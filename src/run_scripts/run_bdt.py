from math import cos, sin, radians
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from ScdmsML.src.utils import data_loader, plot_output
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq", "PTNFdelay"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc", "ptOF", "ptOF0"]
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

fig_save_path = os.path.join("../results/figs/bdt_all_scatters/")
num_scatter_save_path = os.path.join("../results/files/bdt_numscatters.txt")


def do_bdt():
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)

    # BDT time!
    rng = np.random.RandomState(1)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=200)

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
