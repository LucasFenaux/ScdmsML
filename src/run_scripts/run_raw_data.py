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
from src.utils.data_loading import torch_data_loader
from src.models.lstm import LSTMClassifier
from src.main_scripts import train_nn


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
    row_dict = get_all_events(filepaths)
    logging.info("done getting events")
    #logging.info(row_dict.keys())
    logging.info("size of the dict {}".format(sys.getsizeof(row_dict)))
    #with open('raw_data_dict.pickle', 'wb') as handle:
    #    pickle.dump(row_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    matrix = []
    for event in list(row_dict.keys()):
        # Apparently saving each array in its own file seems to be taking too much space
        # also computecanada doesnt like many small files
        # np.save("../../data/raw_events/event_number_{}.npy".format(event), row_dict[event])
        # so we'll append the event number as the first value in the array for each event number and store them all in
        # one big file
        row_dict[event].insert(0, event)
        matrix.append(np.array(row_dict[event]))
    matrix = np.array(matrix)
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

    confusion_matrix = build_confusion_matrix(model, batch_loader, number_of_classes, range(number_of_classes), device)
    confusion_matrix = confusion_matrix.to(torch.device("cpu"))
    # print(np.round(confusion_matrix.numpy()))

    num_samples = sum(confusion_matrix.sum(1))
    correctly_classified = sum(confusion_matrix.diag())

    return correctly_classified / num_samples


if __name__ == "__main__":
    #pre_processing()
    #run_lstm()
