import matplotlib.pyplot as plt
import numpy as np
from .Raw_data import read_file
import pandas as pd
import cdms

# can only be run on a server that has cdms installed

# first gonna write dataloading code here to test it, will move it to data_loading.py once it's working

det = [14]
n_samples = 2048
chan_list = [0, 1, 2, 3, 4, 5]
reindex_const=50000


# TODO: make sure event numbers are the right numbers and do not need adjustment like in the get_full_data function
def get_all_events(filepaths):
    dfs = None
    for idx, filepath in enumerate(filepaths):
        try:
            df = read_file(filepath, detlist=det, chanlist=chan_list, n_samples=n_samples)
        except:
            print("Problems reading dump ", idx)
            print("\t", filepath)
            continue
        if dfs is None:
            dfs = df
        else:
            dfs = pd.concat([dfs, df], axis=0)


if __name__ == '__main__':
    dfs = get_all_events("")
    print(dfs['event number'])

