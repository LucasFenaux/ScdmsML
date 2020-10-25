from __future__ import absolute_import
import numpy as np
import sys
import os
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/src')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# sys.path.insert(0, '/home/lucas/Documents/ScdmsML')
# sys.path.insert(0, '/Users/GeorgesKanaan/Documents/Research/SuperCDMS/SCDMS_ML')
# sys.path.insert(0, '/home/ge0rges/projects/rrg-mdiamond/fenauxlu/ScdmsML')

import torch
import torch.optim as optim
import pickle
import logging
logging.basicConfig(filename='./raw_data_log.log', level=logging.DEBUG)

from src.utils import get_all_events, build_confusion_matrix
from src.utils.data_loading import torch_data_loader, raw_data_loader
from src.models.lstm import LSTMClassifier
from src.main_scripts import train_nn
from src.utils.Raw_data import read_file

num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


def test():
    df = read_file("/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0001.gz")
    df = df.set_index(['event number'])  # , 'channel number'])#,'detector number'])")
    print(df['channel number'].unique())
    print(df.sample(10))


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
        #filepaths.append("../../data/Raw_data/libinput_sb-70V_F0" + last_part)
        filepaths.append("/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)
        #filepaths.append("/home/ge0rges/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)

    logging.info("getting all events")
    matrix = get_all_events(filepaths)
    logging.info("done getting events")
    #logging.info(row_dict.keys())
    logging.info("size of the data matrix {}".format(sys.getsizeof(matrix)))
    #with open('raw_data_dict.pickle', 'wb') as handle:
    #    pickle.dump(row_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # trace starts at index 5, event num is at 0, det is at 1, ev type is at 2, channel num is at 3 and ev cat is at 4
    # we only care about event number
    np.delete(matrix, [1, 2, 3, 4], axis=1)
    np.save("../../data/raw_events/pre_processed_data.npy", matrix)


def run_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")
    num_workers = 4
    batch_size = 256

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 20
    learning_rate = 0.1

    input_size = 0
    hidden_size = 0
    num_layers = 0

    nn = LSTMClassifier(input_size, hidden_size, num_layers).to(device)

    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)

    train_loader, test_loader = torch_raw_data_loader(det=14, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

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

    confusion_matrix = build_confusion_matrix(model, batch_loader, 2, range(2), device)
    confusion_matrix = confusion_matrix.to(torch.device("cpu"))
    # print(np.round(confusion_matrix.numpy()))

    num_samples = sum(confusion_matrix.sum(1))
    correctly_classified = sum(confusion_matrix.diag())

    return correctly_classified / num_samples


if __name__ == "__main__":
    raw_data_loader("/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/data/raw_events/pre_processed_data.npy", "/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_MATLAB_V1-4_PhotoneutronSb/Input_SuperSim/PhotoNeutronDMC_InitialTest10K_jswfix.mat", num_scatter_save_path)
    #pre_processing()
    #run_lstm()
