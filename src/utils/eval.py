import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import logging


def build_confusion_matrix(model, dataloader, number_of_classes, cls, device):
    """ Builds a confusion matrix based on the given model and dataloader.
        https://en.wikipedia.org/wiki/Confusion_matrix"""
    untrained_cls = list(range(number_of_classes))
    for t in cls:
        untrained_cls.remove(t)

    all_binary_outputs = None
    all_binary_targets = None
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        # Go from probabilities to classification
        _, indices = torch.max(outputs, 1)
        binary_outputs = torch.nn.functional.one_hot(indices, num_classes=number_of_classes)
        binary_targets = torch.nn.functional.one_hot(targets, num_classes=number_of_classes)

        # Create the column for all untrained cls if necessary
        if len(untrained_cls) > 0:
            trained_outputs = binary_outputs[:, cls]
            trained_targets = binary_targets[:, cls]

            untrained_outputs = torch.sum(binary_outputs[:, untrained_cls], dim=1).unsqueeze(dim=1)
            untrained_targets = torch.sum(binary_targets[:, untrained_cls], dim=1).unsqueeze(dim=1)

            binary_outputs = torch.cat([trained_outputs, untrained_outputs], dim=1)
            binary_targets = torch.cat([trained_targets, untrained_targets], dim=1)

        if all_binary_outputs is None:
            all_binary_targets = binary_targets
            all_binary_outputs = binary_outputs

        else:
            all_binary_targets = torch.cat([all_binary_targets, binary_targets])
            all_binary_outputs = torch.cat([all_binary_outputs, binary_outputs])

    all_binary_outputs = all_binary_outputs.argmax(1).to(torch.device("cpu"))
    all_binary_targets = all_binary_targets.argmax(1).to(torch.device("cpu"))

    return torch.Tensor(confusion_matrix(all_binary_targets, all_binary_outputs))


def compute_accuracy(predictions, targets):
    """Computes an accuracy based on the given predictions and targets"""
    assert len(predictions) == len(targets)

    accuracy_array = []
    for (idx, pred), (_, t) in zip(enumerate(predictions), enumerate(targets)):
        accuracy_array.append(1) if pred == t else accuracy_array.append(0)

    accuracy = float(sum(accuracy_array))/float(len(accuracy_array))

    return accuracy


def measure_confidence(probabilities, predictions, targets):
    """" Computes the overall confidence of the network, the confidence in its correct choices and
    the confidence in its incorrect choices"""
    # First overall confidence
    overall_confidence = np.average(np.abs((targets - probabilities)))

    # Positive confidence
    indices = np.where(predictions == targets)
    positive_confidence = np.average(np.abs((targets[indices] - probabilities[indices])))

    # Negative confidence
    indices = np.where(predictions != targets)
    negative_confidence = np.average(np.abs((targets[indices] - probabilities[indices])))

    return overall_confidence, positive_confidence, negative_confidence
