from __future__ import absolute_import
import sys
import os
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/src/')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy
import torch
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from src.models.model import LSTMClassifier
from ignite.metrics import Accuracy, Loss
from functools import partial


def ot_func(output):
    y_pred, y = output
    y_pred = torch.nn.functional.one_hot(torch.max(y_pred, 1)[1], num_classes=2).to(torch.float)
    return (y_pred, y)


def setup_event_handler(trainer, evaluator, train_loader, test_loader):
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
        # writer.add_scalar("testing_loss", metrics["nll"], trainer.state.epoch)
        # writer.add_scalar("testing_accuracy", metrics["accuracy"], trainer.state.epoch)


if __name__ == '__main__':
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32

    device = torch.device('cpu')

    input_size = 1
    hidden_size = 8
    label_size = 1
    batch_size = 64
    epochs = 3

    criterion = torch.nn.BCELoss()
    regressor = LSTMClassifier(input_size, hidden_size, label_size, 1, 165)
    regressor = regressor.to(device)
    learning_rate = 0.01
    optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)

    trainer = create_supervised_trainer(regressor, optimizer, criterion, device=device)

    # Setting up the dataset
    train_data = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=1,
                              pin_memory=False)

    trainer.run(train_loader, max_epochs=epochs)

    test_data = torch.Tensor(X_train)
    test_targets = torch.Tensor(y_train)

    test_dataset = TensorDataset(train_data, train_targets)
    test_sampler = RandomSampler(train_dataset)
    test_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=1,
                              pin_memory=False)

    val_metrics = {
        "accuracy": Accuracy(output_transform=partial(ot_func)),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(regressor, metrics=val_metrics, device=device)
    setup_event_handler(trainer, evaluator, train_loader, test_loader)
    trainer.run(train_loader, max_epochs=epochs)
