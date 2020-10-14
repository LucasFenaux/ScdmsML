import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/home/fenauxlu/projects/rrg-mdiamond/fenauxlu/ScdmsML')
# sys.path.insert(0, '/home/lucas/Documents/ScdmsML')
from src.utils import get_all_events
import logging
logging.basicConfig(filename='./raw_data_log.log', level=logging.DEBUG)
import pickle

if __name__ == '__main__':
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
    logging.info("getting all events")
    row_dict = get_all_events(filepaths)
    #trying to save memory
    del filepaths
    logging.info("done getting events")
    #logging.info(row_dict.keys())
    logging.info("size of the dict {}".format(sys.getsizeof(row_dict)))
    #with open('raw_data_dict.pickle', 'wb') as handle:
    #    pickle.dump(row_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for event in list(row_dict.keys()):
        np.save("../../data/raw_events/event_number_{}.npy".format(event), row_dict[event])
