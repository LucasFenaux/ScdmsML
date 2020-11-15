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

from src.utils import get_all_events, build_confusion_matrix, compute_metrics
from src.utils.data_loading import torch_raw_data_loader, raw_data_loader_1
from src.models.lstm import LSTMClassifier
from src.main_scripts import train_nn
from src.utils.Raw_data import read_file
from multiprocessing import cpu_count

num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


# Only run it once to preprocess the data
def pre_processing():
    # First file is data dump, DO NOT INCLUDE IT
    filepaths = []
    for i in range(2, 20):#977):
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
    # we only care about event number and channel
    logging.info("matrix shape before column deletion {}".format(np.shape(matrix)))
    matrix = np.delete(matrix, [1, 2, 4], axis=1)
    logging.info("matrix shape after deletion {}".format(np.shape(matrix)))
    np.save("../../data/raw_events/pre_processed_data.npy", matrix)


def pre_processing_part2():
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    data_3D = []
    for i in range(np.shape(data)[0]):
        row = []
        for j in range(np.shape(data)[1]):
            #train_data[i][j] = np.array([train_data[i][j], 0])
            row.append(np.array([data[i][j]]))
        row = np.array(row)
        data_3D.append(row)
    data = np.array(data_3D)
    np.save("../../data/raw_events/pre_processed_data_3D_1_attribute.npy", data)


def run_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device : {}".format(device))
    assert torch.cuda.is_available()
    pin_memory = (device.type == "cuda")
    # num_cpus = cpu_count()
    # logging.info("Number of CPUs: {}".format(num_cpus))
    num_workers = 8
    batch_size = 1024

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 1000
    learning_rate = 0.0005

    input_size = 1
    hidden_size = 20
    num_layers = 3

    nn = LSTMClassifier(input_size, hidden_size, num_layers, output_dim=1).to(device)

    optimizer = optim.SGD(nn.parameters(), lr=learning_rate)

    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    for i in range(epochs):
        # for param in nn.parameters():
        #     print(param)
        logging.info("epoch {}".format(i))
        loss = train_nn(train_loader, nn, criterion, optimizer, False, device)
        # err = error_function(nn, test_loader)
        # logging.info("Acc: {}".format(err))
        logging.info("Loss: {}".format(loss))
        compute_metrics(nn, test_loader, device)

    # test the model
    loss = train_nn(test_loader, nn, criterion, optimizer, True, device)
    # err = error_function(nn, test_loader)
    logging.info("Final Torch Loss: {}".format(loss))
    compute_metrics(nn, test_loader, device)
    # logging.info("Final Torch Err: {}".format(err))


if __name__ == "__main__":
    #pre_processing()
    #pre_processing_part2()
    #data, targets, target_evs = raw_data_loader_1("/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/data/raw_events/pre_processed_data.npy", "/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_MATLAB_V1-4_PhotoneutronSb/Input_SuperSim/PhotoNeutronDMC_InitialTest10K_jswfix.mat", num_scatter_save_path)
    run_lstm()
