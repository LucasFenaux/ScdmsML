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
from src.utils.misc import get_tensorboard_log_dir
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")
log_dir = get_tensorboard_log_dir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


# Only run it once to preprocess the data
def pre_processing():
    """Preprocessing for the raw data files, it is the same as the run_raw_1_att one"""
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
    """This pre-processing part however is different from the run_raw_1_att one"""
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    data_3D = []
    # keep track of the events already encountered and keep track of their index in the data array
    event_map = {}
    # keep track of which channels have already been added for a particular event as we want
    # each channel to be at the same index for each data sample in the data array
    channel_tracker = {}

    all_event_numbers = data[:, 0]
    all_channel_numbers = data[:, 1]
    for ev_num in all_event_numbers:
        channel_tracker[ev_num] = []
    data = np.delete(data, 0, axis=1)  # remove ev
    data = np.delete(data, 0, axis=1)  # remove channel num

    event_counter = 0  # counter to index the events into the data array

    for i in range(np.shape(data)[0]):
        event = all_event_numbers[i]
        channel = all_channel_numbers[i]

        # see if we encountered that event already
        if event not in list(event_map.keys()):
            event_map[event] = event_counter
            event_counter += 1
        event_idx = event_map[event]

        # we want to keep the channels ordered ascending
        channel_idx = 0
        for j in channel_tracker[event]:


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

    nn = LSTMClassifier(input_size, hidden_size, label_size=1)
    nn = nn.to(device)
    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()

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