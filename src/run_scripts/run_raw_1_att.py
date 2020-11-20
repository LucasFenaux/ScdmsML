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
from src.models.lstm import LSTMClassifier
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from src.utils.misc import get_tensorboard_log_dir
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")
log_dir = get_tensorboard_log_dir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


# Only run it once to preprocess the data
def pre_processing():
    # First file is data dump, DO NOT INCLUDE IT
    filepaths = []
    for i in range(2, 977):
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


def setup_event_handler(trainer, evaluator, train_loader, test_loader):
    log_interval = 10

    writer = SummaryWriter(log_dir=log_dir)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
        writer.add_scalar("training/batch_loss", trainer.state.output, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("training/avg_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("training/avg_accuracy", metrics["accuracy"], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("training/avg_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("training/avg_accuracy", metrics["accuracy"], trainer.state.epoch)


def run():
    num_workers = 8
    batch_size = 1600

    input_size = 1
    hidden_size = 3
    num_layers = 2

    epochs = 500
    learning_rate = 0.005

    assert torch.cuda.is_available()

    nn = LSTMClassifier(input_size, hidden_size, num_layers, output_dim=2).to(device)
    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    run()

