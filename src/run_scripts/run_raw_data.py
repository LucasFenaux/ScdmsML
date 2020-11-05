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

from src.utils import get_all_events, build_confusion_matrix, compute_accuracy, measure_confidence
from src.utils.data_loading import torch_raw_data_loader, raw_data_loader_1
from src.models.lstm import LSTMClassifier
from src.main_scripts import train_nn
from src.utils.Raw_data import read_file
from multiprocessing import cpu_count

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
    num_cpus = cpu_count()
    logging.info("Number of CPUs: {}".format(num_cpus))
    num_workers = 16
    batch_size = 1024

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 1000
    learning_rate = 0.0005

    input_size = 1
    hidden_size = 20
    num_layers = 3

    nn = LSTMClassifier(input_size, hidden_size, num_layers).to(device)

    optimizer = optim.Adam(nn.parameters(), lr=learning_rate)

    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    for i in range(epochs):
        # for param in nn.parameters():
        #     print(param)
        logging.info("epoch {}".format(i))
        loss = train_nn(train_loader, nn, criterion, optimizer, False, device)
        # err = error_function(nn, test_loader)
        # logging.info("Acc: {}".format(err))
        logging.info("Loss: {}".format(loss))
        compute_metrics(nn, test_loader)

    # test the model
    loss = train_nn(test_loader, nn, criterion, optimizer, True, device)
    # err = error_function(nn, test_loader)
    logging.info("Final Torch Loss: {}".format(loss))
    compute_metrics(nn, test_loader)
    # logging.info("Final Torch Err: {}".format(err))


# def error_function(model, batch_loader):
#     """
#     Calculates a metric to judge model. Must return a float.
#     Metric is experiment dependent could be AUROC, Accuracy, Error....
#
#     Metric must be "higher is better" (eg. accuracy)
#
#     Do not modify params. Abstract method for all experiments.
#     """
#
#     confusion_matrix = build_confusion_matrix(model, batch_loader, 2, range(2), device)
#     confusion_matrix = confusion_matrix.to(torch.device("cpu"))
#     # print(np.round(confusion_matrix.numpy()))
#
#     num_samples = sum(confusion_matrix.sum(1))
#     correctly_classified = sum(confusion_matrix.diag())
#
#     return correctly_classified / num_samples


def compute_metrics(model, testloader):
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
        outputs = model(inputs)

        # Go from probabilities to classification
        preds = (outputs + 0.5).to(torch.int64)  # <0.5 goes to 0 and >0.5 goes to 1

        if predictions is None:
            predictions = preds
            targets = t
            probabilities = outputs
        else:
            predictions = torch.cat([predictions, preds])
            targets = torch.cat([targets, t])
            probabilities = torch.cat([probabilities, outputs])

    predictions = predictions.numpy()
    targets = targets.numpy()
    probabilities = probabilities.numpy()

    # We first compute the accuracy of the network
    accuracy = compute_accuracy(predictions, targets)
    logging.info("Accuracy: {}".format(accuracy))

    # Then we look at the confidence of the network in its different choices
    overall_confidence, positive_confidence, negative_confidence = measure_confidence(probabilities, predictions, targets)
    logging.info("Overall confidence: {}".format(overall_confidence))
    logging.info("Positive confidence: {}".format(positive_confidence))
    logging.info("Negative confidence: {}".format(negative_confidence))

    return


if __name__ == "__main__":
    #pre_processing()
    #pre_processing_part2()
    #data, targets, target_evs = raw_data_loader_1("/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/data/raw_events/pre_processed_data.npy", "/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_MATLAB_V1-4_PhotoneutronSb/Input_SuperSim/PhotoNeutronDMC_InitialTest10K_jswfix.mat", num_scatter_save_path)
    run_lstm()
