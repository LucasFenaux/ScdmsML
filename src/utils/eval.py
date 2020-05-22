import torch
from sklearn.metrics import confusion_matrix


def build_confusion_matrix(model, dataloader, number_of_classes, cls, device):
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
        binary_targets = targets

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