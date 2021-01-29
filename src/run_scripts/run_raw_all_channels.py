#from __future__ import absolute_import
print("starting imports")
import numpy as np
import sys
import os
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML/src')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.optim as optim
import logging
logging.basicConfig(filename='./raw_data_log.log', level=logging.WARNING)
from queue import Queue
from threading import Thread
import multiprocessing as mp
import pickle
from src.utils import get_all_events
from src.utils.data_loading import torch_raw_data_loader
from src.models.model import LSTMClassifier
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from src.utils.misc import get_tensorboard_log_dir, NDMinMaxScaler
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")
log_dir = get_tensorboard_log_dir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
print("done with imports")

# Only run it once to preprocess the data
def pre_processing():
    """Preprocessing for the raw data files, it is the same as the run_raw_1_att one"""
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
        # filepaths.append("../../data/Raw_data/libinput_sb-70V_F0" + last_part)
        filepaths.append(
            "/home/fenauxlu/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)
        # filepaths.append("/home/ge0rges/projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)

    logging.info("getting all events")
    matrix = get_all_events(filepaths)
    logging.info("done getting events")
    logging.info("size of the data matrix {}".format(sys.getsizeof(matrix)))
    # we only care about event number and channel
    logging.info("matrix shape before column deletion {}".format(np.shape(matrix)))
    matrix = np.delete(matrix, [1, 2, 4], axis=1)
    logging.info("matrix shape after deletion {}".format(np.shape(matrix)))
    np.save("../../data/raw_events/pre_processed_data.npy", matrix)


def deprecated_pre_processing_part_2():
    """This pre-processing part however is different from the run_raw_1_att one"""
    print("start loading")
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    print("loaded intial pre-processed data")
    data_3D = []
    # keep track of the events already encountered and keep track of their index in the data array
    event_map = {}
    # keep track of indices to be able to map an index in the data array to an event
    index_map = {}
    # keep track of which channels have already been added for a particular event as we want
    # each channel to be at the same index for each data sample in the data array
    channel_tracker = {}

    all_event_numbers = data[:, 0]
    all_channel_numbers = data[:, 1]
    for ev_num in all_event_numbers:
        channel_tracker[ev_num] = []
    data = np.delete(data, 0, axis=1)  # remove ev
    data = np.delete(data, 0, axis=1)  # remove channel num
    print("done setting up and removing ev and channel number from data matrix")
    event_counter = 0  # counter to index the events into the data array
    total_rows = np.shape(data)[0]
    for i in range(total_rows):
        print("{} out of {},  {}%".format(i, total_rows, float(i)/float(total_rows)))
        event = all_event_numbers[i]
        channel = all_channel_numbers[i]

        # see if we encountered that event already
        if event not in list(event_map.keys()):
            event_map[event] = event_counter
            print("event {} added to map".format(event))
            event_counter += 1
        event_idx = event_map[event]

        # we want to keep the channels ordered ascending
        channel_idx = 0
        if channel_tracker[event] is None:
            channel_tracker[event] = []
        assert channel not in channel_tracker[event]
        for j in channel_tracker[event]:
            if channel < j:
                channel_idx += 1
            else:
                break
        channel_tracker[event] = channel_tracker[event].insert(channel_idx, channel)
        if np.shape(data_3D)[0] <= event_idx:
            # this event has no channels in the data yet.
            index_map[event_idx] = event
            event_array = []
            for k in range(np.shape(data)[1]):
                event_array.append([data[i][k]])
            data_3D.append(event_array)
        else:
            for k in range(np.shape(data)[1]):
                data_3D[event_idx][k].insert(channel_idx, data[i][k])
    data_3D = np.array(data_3D)
    np.save("../../data/raw_events/pre_processed_data_3D_all_attribute.npy", data_3D)

    # save the index map for label reconstruction
    with open('../../data/raw_events/index_map.pkl', 'wb') as f:
        pickle.dump(index_map, f, pickle.HIGHEST_PROTOCOL)


def check_format(data):
    """ Function to check the format of the pre-processed data after the first round of pre-processing
    and verify that it is in the format where we have for each event number, all 8 channel outputs in a row in the same
    order across all events"""
    n = np.shape(data)[0]
    all_event_numbers = data[:, 0]
    all_channel_numbers = data[:, 1]
    channels = []
    first_event = None
    current_event = None
    shape_matches = False
    for i in range(n):
        if first_event is None:
            first_event = all_event_numbers[i]
            current_event = first_event
        if current_event != all_event_numbers[i] and i % len(channels) != 0:
            logging.ERROR("data is not in the correct format, event numbers are not order properly, aborting")
            print("data is not in the correct format, event numbers are not order properly, aborting")
            return False, channels, all_channel_numbers[i], i % 8
        if first_event == current_event:
            channels.append(all_channel_numbers[i])
        elif not shape_matches:
            if i % len(channels) != 0:
                logging.ERROR("data is missing some rows")
                print("data is missing some rows")
                return False, [], 0, 0
            else:
                shape_matches = True
        if channels[i%8] != all_channel_numbers[i]:
            logging.ERROR("data is not in the correct format, channel numbers are not order properly, aborting")
            print("data is not in the correct format, channel numbers are not order properly, aborting")
            return False, channels, all_channel_numbers[i], i % 8

    # if we get here, it means the data is in the correct format
    return True, channels, -1, -1


def multi_process_pre_procesing_part_2():
    """ Supposed to do the same thing as the slow pre processing part2, however checks for the format first
    in order to speed up the process. Does require the proper format however"""
    # First check the format
    print("start loading")
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    print("loaded intial pre-processed data")
    is_valid, channel_order, a, b = check_format(data)
    if not is_valid:
        print(str(channel_order), str(a), str(b))
        return

    def multi_process_pre(submatrix_indices):
        submatrix = data[submatrix_indices[0]:submatrix_indices[1], :]
        all_event_numbers = submatrix[:, 0]
        submatrix = np.delete(submatrix, 0, axis=1)  # remove ev
        submatrix = np.delete(submatrix, 0, axis=1)  # remove channel num
        n = int(np.shape(submatrix)[0]/8) # we shouldn't get any rounding error as the check format checks if we have the correct shape
        submatrix_3D = []
        event_numbers = []
        for i in range(n):
            a = submatrix[i:i+8, :]
            a = np.transpose(a)
            submatrix_3D.append(a)
            event_numbers.append(all_event_numbers[8*i])
        submatrix_3D = np.array(submatrix_3D)
        assert np.shape(submatrix_3D)[0] == len(event_numbers)
        return submatrix_3D, event_numbers

    class Worker(Thread):
        def __init__(self, input_queue, output_queue):
            Thread.__init__(self)
            self.input_queue = input_queue
            self.output_queue = output_queue

        def run(self):
            while True:
                indices = self.input_queue.get()
                try:
                    out_submatrix_3D, out_event_numbers = multi_process_pre(indices)
                    self.output_queue.put((out_submatrix_3D, out_event_numbers))
                finally:
                    self.input_queue.task_done()

    in_queue = Queue()
    out_queue = Queue()
    m = mp.cpu_count()
    for x in range(m):
        worker = Worker(in_queue, out_queue)
        worker.daemon = True
        worker.start()
    rows = np.shape(data)[0]
    chunk_size = int(rows/m)
    while chunk_size%8 != 0:
        chunk_size -= 1  # so that we cut on the event separation line and not before
    for y in range(m):
        # create the indices
        if y == m - 1:
            in_queue.put((y*chunk_size, rows))
        else:
            in_queue.put((y*chunk_size, (y+1)*chunk_size))

    # Now we pick up the finished work and reconstruct the array
    data_3D = None
    events = None
    for z in range(m):
        sub_3D, evs = out_queue.get()
        if data_3D is None and events is None:
            data_3D = sub_3D
            events = evs
        else:
            data_3D = np.concatenate(data_3D, sub_3D, axis=0)
            events.extend(evs)
        out_queue.task_done()
    events = np.array(events)
    np.save("../../data/raw_events/pre_processed_data_3D_all_attribute.npy", data_3D)
    np.save("../../data/raw_events/pre_processed_data_events_all_attributes.npy", events)


def normalizing():
    """ Normalizes the pre-processed data """
    data = np.load("../../data/raw_events/pre_processed_data_3D_all_attribute.npy")
    # this pre-processed data does not contain channel or event number anymore
    normalizer = NDMinMaxScaler()
    normalizer.fit(data)
    print(np.shape(data))
    normalized_data = normalizer.transform(data)
    print(np.shape(normalized_data))

    np.save("../../data/raw_events/pre_processed_normalized_data_3D_all_attribute.npy", normalized_data)


def setup_event_handler(trainer, evaluator, train_loader, test_loader):
    log_interval = 10

    writer = SummaryWriter(log_dir=log_dir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.5f}".format(trainer.state.epoch, trainer.state.output))
        writer.add_scalar("training_iteration_loss", trainer.state.output, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("training_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("training_accuracy", metrics["accuracy"], trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED(every=log_interval))
    def log_testing_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Accuracy: {:.5f} Loss: {:.5f}"
                     .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
        writer.add_scalar("testing_loss", metrics["nll"], trainer.state.epoch)
        writer.add_scalar("testing_accuracy", metrics["accuracy"], trainer.state.epoch)


def run():
    num_workers = 8
    batch_size = 512

    input_size = 8
    hidden_size = 20

    epochs = 500
    learning_rate = 0.005

    assert torch.cuda.is_available()

    nn = LSTMClassifier(input_size, hidden_size, label_size=1)
    nn = nn.to(device)
    train_loader, test_loader = torch_raw_data_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    optimizer = optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=0.001)

    criterion = torch.nn.BCELoss()

    trainer = create_supervised_trainer(nn, optimizer, criterion, device=device)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(nn, metrics=val_metrics, device=device)

    setup_event_handler(trainer, evaluator, train_loader, test_loader)

    trainer.run(train_loader, max_epochs=epochs)


def test_function():
    print("start loading")
    data = np.load("../../data/raw_events/pre_processed_data.npy")
    data_3D = []
    # keep track of the events already encountered and keep track of their index in the data array
    event_map = {}
    # keep track of indices to be able to map an index in the data array to an event
    index_map = {}
    # keep track of which channels have already been added for a particular event as we want
    # each channel to be at the same index for each data sample in the data array
    channel_tracker = {}

    all_event_numbers = data[:, 0]
    all_channel_numbers = data[:, 1]
    for ev_num in all_event_numbers:
        channel_tracker[ev_num] = []
    data = np.delete(data, 0, axis=1)  # remove ev
    data = np.delete(data, 0, axis=1)  # remove channel num
    print("done setting up and removing ev and channel number from data matrix")
    event_counter = 0  # counter to index the events into the data array
    total_rows = np.shape(data)[0]
    print("total_rows: {}".format(total_rows))
    for i in range(total_rows):
        event = all_event_numbers[i]
        channel = all_channel_numbers[i]
        # see if we encountered that event already
        if event not in list(event_map.keys()):
            event_map[event] = event_counter
            print("event {} added to map".format(event))
            event_counter += 1
        event_idx = event_map[event]

    print("done setting up the matrix")    
    
if __name__ == '__main__':
    print("program loaded, starting pre-processing")
    pre_processing_part_2()
    print("done with pre-processing, starting normalization")
#    normalizing()
    # run()
