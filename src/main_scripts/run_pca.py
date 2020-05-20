from math import cos, sin, radians
import numpy as np
from ScmdsML.src.utils import data_loader
from sklearn.decomposition import PCA
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq"]
rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
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


if __name__ == '__main__':
    n_comp, variances = do_pca(threshold=0.99)
