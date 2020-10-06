import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.insert(0, '/home/fenauxlu/ScdmsML')
sys.path.insert(0, '/home/lucas/Documents/ScdmsML')
from src.utils import get_all_events


if __name__ == '__main__':
    # First file is data dump, DO NOT INCLUDE IT
    filepaths = []
    for i in range(2, 500):
        last_part = ""
        if i >= 100:
            last_part += str(i)
        elif i >= 10:
            last_part = last_part + "0" + str(i)
        else:
            last_part = last_part + "00" + str(i)
        last_part += ".gz"
        filepaths.append("../../data/Raw_data/libinput_sb-70V_F0" + last_part)
        # filepaths.append("../../../projects/rrg-mdiamond/data/Soudan/DMC_V1-5_PhotoneutronSb/Raw/libinput_sb-70V_F0" + last_part)
    row_dict = get_all_events(filepaths)
    print(row_dict.keys())

