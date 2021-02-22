#from __future__ import absolute_import
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
from queue import Queue
from threading import Thread
import multiprocessing as mp
import pickle
from src.utils import get_all_events
from src.utils.data_loading import torch_all_channels_raw_data_loader
from src.models.model import LSTMClassifier, CNN_LSTM_Classifier
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from src.utils.misc import get_tensorboard_log_dir, NDMinMaxScaler
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import csv
log_dir = get_tensorboard_log_dir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
from functools import partial
# traindatafilepath = '../../data/UCRArchive_2018/SyntheticControl/SyntheticControl_TRAIN.tsv'
# testdatafilepath = '../../data/UCRArchive_2018/SyntheticControl/SyntheticControl_TEST.tsv'
traindatafilepath = '../../data/UCRArchive_2018/PigAirwayPressure/PigAirwayPressure_TRAIN.tsv'
testdatafilepath = '../../data/UCRArchive_2018/PigAirwayPressure/PigAirwayPressure_TEST.tsv'
batch_size = 1000
num_workers = 3
num_classes = 52


def dataloading():
    with open(traindatafilepath, newline='') as csvfile:
        train_data = list(csv.reader(csvfile, delimiter='\t'))

    train_data = np.array(train_data, dtype=np.float64)
    print(np.shape(train_data))
    # the labels are on the first column
    train_labels = train_data[:, 0] - 1 # need to have the labels start at 0, not 1
    train_data = train_data[:, 1:]
    train_data = np.reshape(train_data, (np.shape(train_data)[0], np.shape(train_data)[1], 1))

    with open(testdatafilepath, newline='') as csvfile:
        test_data = list(csv.reader(csvfile, delimiter='\t'))

    test_data = np.array(test_data, dtype=np.float64)
    # the labels are on the first column
    test_labels = test_data[:, 0] - 1 # need to have the labels start at 0, not 1
    test_data = test_data[:, 1:]
    test_data = np.reshape(test_data, (np.shape(test_data)[0], np.shape(test_data)[1], 1))

    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_labels)
    if num_classes == 2:
        # we only one-hot the targets for binary cross entropy
        train_targets = torch.nn.functional.one_hot(train_targets.to(torch.int64))
    else:
        train_targets = train_targets.to(torch.int64)
    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_labels)
    if num_classes == 2:
        # we only one-hot the targets for binary cross entropy
        test_targets = torch.nn.functional.one_hot(test_targets.to(torch.int64))
    else:
        test_targets = test_targets.to(torch.int64)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def setup_event_handler(trainer, evaluator, train_loader, test_loader):
    log_interval = 10

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


def run_cnnlstm():
    input_size = 1
    hidden_size = 100

    epochs = 1000
    learning_rate = 0.001
    dropout_rate = 0.2

    # assert torch.cuda.is_available()

    nn = CNN_LSTM_Classifier(input_size, hidden_size, label_size=num_classes, device=device, dropout_rate=dropout_rate)
    nn = nn.to(device)
    train_loader, test_loader = dataloading()
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=0.0001)
    if num_classes == 2:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=num_classes).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader)

    trainer.run(train_loader, max_epochs=epochs)


def run_lstm():
    input_size = 1
    hidden_size = 20

    epochs = 1000
    learning_rate = 0.001
    dropout_rate = 0.2

    # assert torch.cuda.is_available()

    nn = LSTMClassifier(input_size, hidden_size, label_size=num_classes, device=device, dropout_rate=dropout_rate)
    nn = nn.to(device)
    train_loader, test_loader = dataloading()
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=0.0001)
    if num_classes == 2:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=num_classes).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    run_cnnlstm()
