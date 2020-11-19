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


def compute_metrics(model, testloader, device):
    """
    Compute multiple metrics based on the given model and the test loader.

    :param model: given model
    :param testloader:  test loader
    :return: nothing
    """

    # First get the model's predictions for the test data
    predictions = None
    targets = None
    probabilities = None

    for i, (inputs, t) in enumerate(testloader):
        inputs = inputs.to(device)
        t = t.to(device)
        outputs = model(inputs)#.to(torch.device("cpu"))
        #logging.info("{}".format(i))
        # If both targets and outputs are 1D Go from probabilities to classification
        #preds = (outputs + 0.5).to(torch.device("cpu")).to(torch.int64)  # <0.5 goes to 0 and >0.5 goes to 1

        # Else

        # If edelting the variables doesnt work, then try detaching before
        # doing the computations
        inputs = inputs.detach().to(torch.device("cpu"))
        t = t.detach().to(torch.device("cpu"))
        outputs = outputs.detach().to(torch.device("cpu"))
        _, preds = torch.max(outputs, 1)

        preds = torch.nn.functional.one_hot(preds, num_classes=2)
        t = torch.nn.functional.one_hot(t, num_classes=2)
        #t = t.to(torch.device("cpu"))
        if predictions is None:
            predictions = preds
            targets = t
            probabilities = outputs
        else:
            predictions = torch.cat([predictions, preds])
            targets = torch.cat([targets, t])
            probabilities = torch.cat([probabilities, outputs])#.to(torch.device("cpu"))])
        del t
        del preds
        del outputs
        del inputs
        torch.cuda.empty_cache()

    predictions = predictions.to(torch.device("cpu")).detach().numpy()
    targets = targets.to(torch.device("cpu")).detach().numpy()
    probabilities = probabilities.to(torch.device("cpu")).detach().numpy()

    torch.cuda.empty_cache()

    #logging.info("predictions : {}".format(np.shape(predictions)))
    #logging.info("targets : {}".format(np.shape(targets)))
    #logging.info("probabilities : {}".format(np.shape(probabilities)))

    # We first compute the accuracy of the network
    accuracy = compute_accuracy(predictions, targets)
    # logging.info("Accuracy: {}".format(accuracy))

    # Then we look at the confidence of the network in its different choices
    overall_confidence, positive_confidence, negative_confidence = measure_confidence(probabilities, predictions, targets)
    logging.info("Overall confidence: {}".format(overall_confidence))
    logging.info("Positive confidence: {}".format(positive_confidence))
    logging.info("Negative confidence: {}".format(negative_confidence))
    return accuracy


def compute_accuracy(predictions, targets):
    """Computes an accuracy based on the given predictions and targets"""
    assert len(predictions) == len(targets)

    accuracy_array = []

    if len(np.shape(targets)) == 1:

        for (idx, pred), (_, t) in zip(enumerate(predictions), enumerate(targets)):
            accuracy_array.append(1) if pred == t else accuracy_array.append(0)

    else:

        for (idx, pred), (_, t) in zip(enumerate(predictions), enumerate(targets)):
            accuracy_array.append(1) if np.argmax(pred) == np.argmax(t) else accuracy_array.append(0)

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
