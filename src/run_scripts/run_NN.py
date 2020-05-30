import torch
import torch.optim as optim
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ScdmsML.src.models import NeuralNetwork
from ScdmsML.src.main_scripts import train_nn
from ScdmsML.src.utils import torch_data_loader, build_confusion_matrix, data_loader

from math import cos, sin, radians
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq", "PTNFdelay"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc"]
rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
                 , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
                 "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc", "ptOF", "ptOF0",
                 "ptOF1X2P", "ptOF1X2P0", "ptOF1X2P0uc", "ptOF1X2Puc", "ptOF1X2R", "ptOF1X2R0", "ptftWK_9520",
                 "pxpartOF"]
# rrq_var_names = ['DetType', 'Empty', 'pa2tdelr', 'paOF', 'paOF0', 'paOF1X2P', 'paOF1X2P0', 'paOF1X2R', 'paOF1X2R0',
#                  'paampres', 'padelayres', 'pb2tdelr', 'pbOF', 'pbOF0', 'pbOF1X2P', 'pbOF1X2P0', 'pbOF1X2R', 'pbOF1X2R0'
#                  , 'pbampres', 'pbdelayres', 'pc2tdelr', 'pcOF', 'pcOF0', 'pcOF1X2P', 'pcOF1X2P0', 'pcOF1X2R',
#                  'pcOF1X2R0', 'pcampres', 'pcdelayres', 'pd2tdelr', 'pdOF', 'pdOF0', 'pdOF1X2P', 'pdOF1X2P0',
#                  'pdOF1X2R', 'pdOF1X2R0', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
#                  'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF', 'pprimechanOFWK'
#                  , 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prpartOF1X2', 'prxypartOF', 'prxypartOF1X2',
#                  'psumOF', 'psumOF1X2', 'psumiOF', 'psumiOF1X2', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0uc', 'ptNFuc', 'ptOF'
#                  , 'ptOF0', 'ptOF0uc', 'ptOF1X2P', 'ptOF1X2P0', 'ptOF1X2P0uc', 'ptOF1X2Puc', 'ptOF1X2R', 'ptOF1X2R0',
#                  'ptOFuc', 'ptftWK_9520', 'pxdelWK', 'pxpartOF', 'pxpartOF1X2', 'pydelWK', 'pypartOF', 'pypartOF1X2',
#                  'qiOF', 'qiOF0', 'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']
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

criterion = torch.nn.MSELoss()  # Change to use different loss function


def train_torch_model(pca=0):
    epochs = 30
    learning_rate = 0.01
    momentum = 0.9
    if pca != 0:
        sizes = [pca, 10, 10, 2]
    else:
        sizes = [31, 30, 30, 2]

    nn = NeuralNetwork(sizes=sizes).to(device)

    optimizer = optim.SGD(nn.parameters(), lr=learning_rate, momentum=momentum)

    train_loader, test_loader = torch_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path,
                                                  det=14, batch_size=batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory, with_pca=pca)

    for _ in range(epochs):
        # for param in nn.parameters():
        #     print(param)
        loss = train_nn(train_loader, nn, criterion, optimizer, False, device)
        err = error_function(nn, test_loader)
        print("Err: ", err)

    # test the model
    loss = train_nn(test_loader, nn, criterion, optimizer, True, device)
    err = error_function(nn, test_loader)
    print("Final Torch Loss: ", loss)
    print("Final Torch Err: ", err)


def train_sklearn_model():
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)

    model = MLPClassifier(hidden_layer_sizes=(30, 30), solver="adam", activation="relu", learning_rate="constant"
                          , max_iter=100, n_iter_no_change=5).fit(train_data, train_targets)
    print(model.predict_proba(test_data))
    acc = model.score(test_data, test_targets)
    print("Sklearn acc:", acc)


def error_function(model, batch_loader):
    """
    Calculates a metric to judge model. Must return a float.
    Metric is experiment dependent could be AUROC, Accuracy, Error....

    Metric must be "higher is better" (eg. accuracy)

    Do not modify params. Abstract method for all experiments.
    """

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_classes, range(number_of_classes), device)
    confusion_matrix = confusion_matrix.to(torch.device("cpu"))
    print(np.round(confusion_matrix.numpy()))

    num_samples = sum(confusion_matrix.sum(1))
    correctly_classified = sum(confusion_matrix.diag())

    return correctly_classified / num_samples


if __name__ == '__main__':
    train_torch_model(6)
    # train_sklearn_model()
