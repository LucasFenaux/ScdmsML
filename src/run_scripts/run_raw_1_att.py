from __future__ import absolute_import
import numpy as np
import sys
import os
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/src')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.optim as optim
import logging
logging.basicConfig(filename='./raw_data_log.log', level=logging.WARNING)

from src.utils import get_all_events
from src.utils.data_loading import torch_raw_data_loader
from src.models.model import LSTMClassifier
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from src.utils.misc import get_tensorboard_log_dir, NDMinMaxScaler
from functools import partial

num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")
log_dir = get_tensorboard_log_dir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


# Only run it once to preprocess the data
def pre_processing():
    """Preprocessing for the raw data files, it is the same as the run_raw_all_channels one"""
    # First file is data dump, DO NOT INCLUDE IT
    filepaths = []
    for i in range(2,977):
        last_part = ""
        if i >= 100:
            last_part += str(i)
        elif i >= 10:
            last_part = last_part + "0" + str(i)
        else:
            last_part = last_part + "00" + str(i)
        last_part += ".gz"
        # filepaths.append("../../data/Raw_data/libinput_sb-70V_F0" + last_part)
        filepaths.append(
            "/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)
        # filepaths.append("/home/ge0rges/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)

    logging.info("getting all events")
    matrix = get_all_events(filepaths)
    logging.info("done getting events")
    logging.info("size of the data matrix {}".format(sys.getsizeof(matrix)))
    # we only care about event number and channel
    logging.info("matrix shape before column deletion {}".format(np.shape(matrix)))
    matrix = np.delete(matrix, [1, 2, 4], axis=1)
    logging.info("matrix shape after deletion {}".format(np.shape(matrix)))
    np.save("../../data/raw_events/pre_processed_data.npy", matrix)


def pre_processing_part2():
    """This pre-processing part however is different from the run_raw_all_channels one"""
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    data_3D = []
    for i in range(np.shape(data)[0]):
        row = []
        for j in range(np.shape(data)[1]):
            row.append(np.array([data[i][j]]))
        row = np.array(row)
        data_3D.append(row)
    data = np.array(data_3D)
    np.save("../../data/raw_events/pre_processed_data_3D_1_attribute.npy", data)


def normalizing():
    """ Normalizes the pre-processed data """
    data = np.load("../../data/raw_events/pre_processed_data_3D_1_attribute.npy")
    # remove event numbers and channel numbers
    all_event_numbers = data[:, 0]
    data = np.delete(data, 0, axis=1)
    all_channel_numbers = data[:, 1]
    data = np.delete(data, 0, axis=1)

    normalizer = NDMinMaxScaler()
    normalizer.fit(data)
    print('Min: %f, Max: %f' % (normalizer.data_min_, normalizer.data_max_))
    normalized_data = normalizer.transform(data)
    # re-insert event number and channel number
    normalized_data = np.insert(normalized_data, 0, all_channel_numbers, axis=0)
    normalized_data = np.insert(normalized_data, 0, all_event_numbers, axis=0)

    np.save("../../data/raw_events/pre_processed_normalized_data_3D_1_attribute.npy", normalized_data)


def setup_event_handler(trainer, evaluator, train_loader, test_loader):
    log_interval = 25

    writer = SummaryWriter(log_dir=log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.5f}".format(trainer.state.epoch, trainer.state.output))
        writer.add_scalar("training_iteration_loss", trainer.state.output, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("training_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("training_accuracy", metrics["accuracy"], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_testing_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("testing_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("testing_accuracy", metrics["accuracy"], trainer.state.epoch)


def run():
    num_workers = 8
    batch_size = 1024
    dropout_rate = 0.1
    input_size = 1
    hidden_size = 50

    epochs = 1000
    
    learning_rate = 0.01  # 0.005, 0.001, 0.1

    assert torch.cuda.is_available()

    nn = LSTMClassifier(input_size, hidden_size, label_size=2, dropout_rate=dropout_rate)
    nn = nn.to(device)
    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)

    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.NLLLoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }

    # lb = lambda x, y, y_pred: (torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float), y)
    #transform = transforms.Lambda(lb)
#SBATCH --gres=grpu:1
    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    run()
    #pre_processing()
    #pre_processing_part2()

