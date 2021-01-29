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
logging.basicConfig(filename='./raw_data_log.log', level=logging.INFO)

from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from sklearn.model_selection import train_test_split
from src.models.model import LSTMClassifier, BiLSTMClassifier, FFClassifier
from src.utils.misc import CustomTracker
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from functools import partial
import random
random.seed(111)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pin_memory = (device.type == "cuda")
device = torch.device("cpu")
pin_memory = False


def level_1_multiple_loader(batch_size=64, num_workers=1, sequence_length=150, pin_memory=False):
    """ Generates sequence of numbers based on some equations with some truth values """
    # Select a random number as seed for the algorithm
    r = random.randint(1, 100)

    # Now we generate n true samples and n false samples
    n = 1000
    # With sequence length of l
    l = sequence_length
    samples = []
    labels = []
    for i in range(n):
        sample = []
        label = None
        if random.randint(0, 1) == 0:
            # We create a sample with false label
            # We make sure it has at least 1 element that is not a multiple of r
            num = r*random.randint(1, 100) + 1
            rand_idx = random.randint(0, 149)
            for j in range(l):
                sample.append(np.array([random.randint(1, 10000)]))
            sample[rand_idx] = np.array([num])
            label = 0
        else:
            for j in range(l):
                # We create a sequence where all the numbers are multiples of r
                k = random.randint(1, 100)
                sample.append(np.array([k*r]))
            label = 1
        sample = np.array(sample)
        samples.append(sample)
        labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)

    train_data, test_data, train_targets, test_targets = train_test_split(samples, labels)
    print("train data shape {}".format(np.shape(train_data)))
    print("test data shape {}".format(np.shape(test_data)))
    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_targets).to(torch.int64)
    train_targets = torch.nn.functional.one_hot(train_targets).to(torch.float)
    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_targets).to(torch.int64)
    test_targets = torch.nn.functional.one_hot(test_targets).to(torch.float)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def test_level_1_multiple_loader(batch_size=64, num_workers=1, sequence_length=150, pin_memory=False):
    """ Generates sequence of numbers based on some equations with some truth values """
    # Select a random number as seed for the algorithm
    r = random.randint(1, 100)

    # Now we generate n true samples and n false samples
    n = 1000
    # With sequence length of l
    l = sequence_length
    samples = []
    labels = []
    for i in range(n):
        sample = []
        label = None
        if random.randint(0, 1) == 0:
            # We create a sample with false label
            # We make sure it has at least 1 element that is not a multiple of r
            num = r*random.randint(1, 100) + 1
            rand_idx = random.randint(0, 149)
            for j in range(l):
                sample.append(np.array(random.randint(1, 10000)))
            sample[rand_idx] = np.array([num])
            label = 0
        else:
            for j in range(l):
                # We create a sequence where all the numbers are multiples of r
                k = random.randint(1, 100)
                sample.append(np.array(k*r))
            label = 1
        sample = np.array(sample)
        samples.append(np.array([sample, sample]))
        labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)

    train_data, test_data, train_targets, test_targets = train_test_split(samples, labels)
    print("train data shape {}".format(np.shape(train_data)))
    print("test data shape {}".format(np.shape(test_data)))
    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_targets).to(torch.int64)
    train_targets = torch.nn.functional.one_hot(train_targets).to(torch.float)
    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_targets).to(torch.int64)
    test_targets = torch.nn.functional.one_hot(test_targets).to(torch.float)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def ff_level_1_multiple_loader(batch_size=64, num_workers=1, sequence_length=150, pin_memory=False):
    """ Generates sequence of numbers based on some equations with some truth values """
    # Select a random number as seed for the algorithm
    r = random.randint(1, 100)

    # Now we generate n true samples and n false samples
    n = 1000
    # With sequence length of l
    l = sequence_length
    samples = []
    labels = []
    for i in range(n):
        sample = []
        label = None
        if random.randint(0, 1) == 0:
            # We create a sample with false label
            # We make sure it has at least 1 element that is not a multiple of r
            num = r*random.randint(1, 100) + 1
            rand_idx = random.randint(0, 149)
            for j in range(l):
                sample.append(random.randint(1, 10000))
            sample[rand_idx] = num
            label = 0
        else:
            for j in range(l):
                # We create a sequence where all the numbers are multiples of r
                k = random.randint(1, 100)
                sample.append(k*r)
            label = 1
        sample = np.array(sample)
        samples.append(sample)
        labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)

    train_data, test_data, train_targets, test_targets = train_test_split(samples, labels)
    print("train data shape {}".format(np.shape(train_data)))
    print("test data shape {}".format(np.shape(test_data)))
    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_targets).to(torch.int64)
    train_targets = torch.nn.functional.one_hot(train_targets).to(torch.float)
    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_targets).to(torch.int64)
    test_targets = torch.nn.functional.one_hot(test_targets).to(torch.float)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def level_2_multiple_loader(batch_size=64, num_workers=1, sequence_length=150, pin_memory=False):
    """ Generates sequence of numbers based on some equations with some truth values """
    # Select a random number as seed for the algorithm
    r = random.randint(1, 100)

    # Now we generate n true samples and n false samples
    n = 1000
    # With sequence length of l
    l = sequence_length
    samples = []
    labels = []
    for i in range(n):
        sample = []
        label = None
        u = random.randint(0, 3)
        if u == 0 or u == 1:
            # We create a sample with false label
            # We make sure it has at least 1 element that is not a multiple of r
            num = r*random.randint(1, 100) + 1
            rand_idx = random.randint(0, 149)
            for j in range(l):
                sample.append(np.array([random.randint(1, 10000)]))
            sample[rand_idx] = np.array([num])
            label = 0
        elif u == 2:
            for j in range(l):
                # We create a sequence where all the numbers are multiples of r
                k = random.randint(1, 100)
                sample.append(np.array([k*r]))
            label = 1
        else:
            # We create a sample with false label
            # We make sure it has at least 1 element that is not a multiple of r
            num = r*random.randint(1, 100) + 1
            rand_idx = random.randint(0, 149)
            for j in range(l):
                # We create a sequence where all the numbers are multiples of r
                k = random.randint(1, 100)
                sample.append(np.array([k*r]))
            sample[rand_idx] = np.array([num])
            label = 0
        sample = np.array(sample)
        samples.append(sample)
        labels.append(label)
    samples = np.array(samples)
    labels = np.array(labels)

    train_data, test_data, train_targets, test_targets = train_test_split(samples, labels)
    print("train data shape {}".format(np.shape(train_data)))
    print("test data shape {}".format(np.shape(test_data)))
    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_targets).to(torch.int64)
    train_targets = torch.nn.functional.one_hot(train_targets).to(torch.float)
    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_targets).to(torch.int64)
    test_targets = torch.nn.functional.one_hot(test_targets).to(torch.float)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, test_loader


def setup_event_handler(trainer, evaluator, train_loader, test_loader, custom_tracker):
    log_interval = 3

    # writer = SummaryWriter(log_dir=log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.5f}".format(trainer.state.epoch, trainer.state.output))
        # writer.add_scalar("training_iteration_loss", trainer.state.output, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
              .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        # writer.add_scalar("training_loss", metrics["nll"], trainer.state.epoch)
        # writer.add_scalar("training_accuracy", metrics["accuracy"], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_testing_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
              .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        custom_tracker.assessment()
        custom_tracker.update_best_accuracy(metrics["accuracy"])
        # writer.add_scalar("testing_loss", metrics["nll"], trainer.state.epoch)
        # writer.add_scalar("testing_accuracy", metrics["accuracy"], trainer.state.epoch)


def run_bilstm():
    num_workers = 2
    batch_size = 128

    input_size = 1
    hidden_size = 10
    sequence_length = 2400

    epochs = 300

    learning_rate = 0.005  # 0.005, 0.001, 0.1

    nn = BiLSTMClassifier(input_size, hidden_size, label_size=2, device=device, dropout_rate=0.2)
    nn = nn.to(device)
    train_loader, test_loader = level_1_multiple_loader(batch_size=batch_size, num_workers=num_workers,
                                                        sequence_length=sequence_length, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=0.001)
    custom_tracker = CustomTracker(nn, device, test_loader, 0)
    criterion = torch.nn.BCELoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader, custom_tracker)

    trainer.run(train_loader, max_epochs=epochs)

    print("best accuracy: ", custom_tracker.get_best_accuracy())


def run_lstm():
    num_workers = 2
    batch_size = 128

    input_size = 1
    hidden_size = 10
    sequence_length = 2400

    epochs = 300

    learning_rate = 0.005  # 0.005, 0.001, 0.1

    nn = LSTMClassifier(input_size, hidden_size, label_size=2, device=device, dropout_rate=0.2)
    nn = nn.to(device)
    train_loader, test_loader = level_1_multiple_loader(batch_size=batch_size, num_workers=num_workers,
                                                        sequence_length=sequence_length, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=0.001)
    custom_tracker = CustomTracker(nn, device, test_loader, 0)
    criterion = torch.nn.BCELoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader, custom_tracker)

    trainer.run(train_loader, max_epochs=epochs)

    print("best accuracy: ", custom_tracker.get_best_accuracy())


def run_ff():
    num_workers = 2
    batch_size = 250

    sequence_length = 4096

    epochs = 1000

    learning_rate = 0.1  # 0.005, 0.001, 0.1

    nn = FFClassifier(dropout_rate=0.5)
    nn = nn.to(device)
    train_loader, test_loader = ff_level_1_multiple_loader(batch_size=batch_size, num_workers=num_workers,
                                                           sequence_length=sequence_length, pin_memory=pin_memory)
    optimizer = optim.SGD(nn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    custom_tracker = CustomTracker(nn, device, test_loader, 0)
    criterion = torch.nn.BCELoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    def ot_func(output):
        y_pred, y = output
        y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float)
        return (y_pred, y)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader, custom_tracker)

    trainer.run(train_loader, max_epochs=epochs)

    print("best accuracy: ", custom_tracker.get_best_accuracy())


if __name__ == '__main__':
    # run_ff()
    run_lstm()
