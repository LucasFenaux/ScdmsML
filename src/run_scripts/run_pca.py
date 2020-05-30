from math import cos, sin, radians
import numpy as np
from ScdmsML.src.utils import data_loader
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc", "ptOF", "ptOF0",
#                  "ptOF1X2P", "ptOF1X2P0", "ptOF1X2P0uc", "ptOF1X2Puc", "ptOF1X2R", "ptOF1X2R0", "ptftWK_9520",
#                  "pxpartOF"]
rrq_var_names = ['DetType', 'Empty', 'pa2tdelr', 'paOF', 'paOF0', 'paOF1X2P', 'paOF1X2P0', 'paOF1X2R', 'paOF1X2R0',
                 'paampres', 'padelayres', 'pb2tdelr', 'pbOF', 'pbOF0', 'pbOF1X2P', 'pbOF1X2P0', 'pbOF1X2R', 'pbOF1X2R0'
                 , 'pbampres', 'pbdelayres', 'pc2tdelr', 'pcOF', 'pcOF0', 'pcOF1X2P', 'pcOF1X2P0', 'pcOF1X2R',
                 'pcOF1X2R0', 'pcampres', 'pcdelayres', 'pd2tdelr', 'pdOF', 'pdOF0', 'pdOF1X2P', 'pdOF1X2P0',
                 'pdOF1X2R', 'pdOF1X2R0', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
                 'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF', 'pprimechanOFWK'
                 , 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prpartOF1X2', 'prxypartOF', 'prxypartOF1X2',
                 'psumOF', 'psumOF1X2', 'psumiOF', 'psumiOF1X2', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0uc', 'ptNFuc', 'ptOF'
                 , 'ptOF0', 'ptOF0uc', 'ptOF1X2P', 'ptOF1X2P0', 'ptOF1X2P0uc', 'ptOF1X2Puc', 'ptOF1X2R', 'ptOF1X2R0',
                 'ptOFuc', 'ptftWK_9520', 'pxdelWK', 'pxpartOF', 'pxpartOF1X2', 'pydelWK', 'pypartOF', 'pypartOF1X2',
                 'qiOF', 'qiOF0', 'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']
new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")


def do_pca(threshold=0.96, proportion=0.):
    assert 0 <= proportion <= 1
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)
    print(np.shape(train_data))
    print("Doing PCA")
    if proportion != 0:
        selected_train_data = []
        print(np.shape(train_data))
        n = np.shape(train_data)[0]
        indices = np.random.choice(list(range(n)), size=(int(n * proportion)))
        inputs = np.array(train_data)
        data = inputs[indices]
        selected_train_data.extend(data)
    else:
        selected_train_data = train_data

    selected_train_data = np.array(selected_train_data)
    model = PCA()
    model.fit_transform(selected_train_data)
    variance_cumsum = model.explained_variance_ratio_.cumsum()

    n_comp = 0
    variances = []
    for cum_variance in variance_cumsum:
        variances.append(cum_variance)
        if cum_variance >= threshold:
            n_comp += 1
            break
        else:
            n_comp += 1

    print("Done with PCA, got:", n_comp)
    print("Variances: ", variances)

    return n_comp, variances


def get_covariance():
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)
    cov = EmpiricalCovariance().fit(train_data)
    for i in range(np.shape(cov.covariance_)[0]):
        print(cov.covariance_[i])

    print("\n", cov.location_)


if __name__ == '__main__':
    n_comp, variances = do_pca(threshold=0.99)
    # get_covariance()
