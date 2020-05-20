from math import cos, sin, radians
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from ScmdsML.src.utils import data_loader, plot_output
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

fig_save_path = os.path.join("../results/figs/bdt_all_scatters/")
num_scatter_save_path = os.path.join("../results/files/bdt_numscatters.txt")


def do_bdt():
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)

    # BDT time!
    rng = np.random.RandomState(1)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=200)

    bdt.fit(train_data, train_targets)
    
    print(bdt.feature_importances_)
    
    predictions = bdt.predict(test_data)
    scores = bdt.decision_function(test_data)
    num_tests = len(scores)
    
    split_scores = [[], []]
    for i in range(len(predictions)):
        split_scores[test_targets[i]].append(scores[i])
    
    return scores, split_scores, test_data, test_targets, test_dict, variables[0].keys()


if __name__ == '__main__':
    scores, split_scores, test_matrix, test_targets, test_dict, var_names = do_bdt()
    print(var_names)
    plot_output(scores, split_scores, test_matrix, test_targets, test_dict, fig_save_path)
