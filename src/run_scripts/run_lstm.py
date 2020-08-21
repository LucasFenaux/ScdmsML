import torch
import torch.nn as nn
import torch.optim as optim
from ScdmsML.src.models import LSTMClassifier
from ScdmsML.src.main_scripts import train_nn
from ScdmsML.src.utils import torch_data_loader, build_confusion_matrix

from math import cos, sin, radians
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq"]
rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/nn_numscatters.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 4
batch_size = 256
number_of_classes = 2

criterion = torch.nn.CrossEntropyLoss()  # Change to use different loss function


def train_torch_model():
    epochs = 20
    learning_rate = 0.1

    input_size = 7
    hidden_size = 7
    num_layers = 1

    nn = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)

    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)

    train_loader, test_loader = torch_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path,
                                                  det=14, batch_size=batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)
    
    for _ in range(epochs):
        # for param in nn.parameters():
        #     print(param)
        loss = train_nn(train_loader, nn, criterion, optimizer, False, device)
        err = error_function(nn, test_loader)
        print("Err: ", err)
        print("Loss: ", loss)

    # test the model
    loss = train_nn(test_loader, nn, criterion, optimizer, True, device)
    err = error_function(nn, test_loader)
    print("Final Torch Loss: ", loss)
    print("Final Torch Err: ", err)


def error_function(model, batch_loader):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Metric must be "higher is better" (eg. accuracy)

    Do not modify params. Abstract method for all experiments.
    """

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_classes, range(number_of_classes), device)
    confusion_matrix = confusion_matrix.to(torch.device("cpu"))
    # print(np.round(confusion_matrix.numpy()))

    num_samples = sum(confusion_matrix.sum(1))
    correctly_classified = sum(confusion_matrix.diag())

    return correctly_classified / num_samples


if __name__ == '__main__':
    train_torch_model()
