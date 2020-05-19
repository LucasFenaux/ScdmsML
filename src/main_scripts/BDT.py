from math import cos, sin, radians
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from ScmdsML.src.utils import *

# Global Variable
det = 14
rqvarnames = ["PTNFchisq"]
rrqvarnames = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
newvarnames = ["PXTFPchisq", "PYTFPchisq"]
newvarinputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
newvarfuncs = [lambda args: (cos(radians(30))*args[0] + cos(radians(150))*args[1] + cos(radians(270))*args[2]),
               lambda args: (sin(radians(30))*args[0] + sin(radians(150))*args[1] + sin(radians(270))*args[2])]
newvarinfo = {"names": [], "inputs": [], "funcs": []}

calibpath = "../data/calib_LibSimProdv5-4_pn_Sb_T5Z2.root"
mergepath = "../data/merge_LibSimProdv5-4_pn_Sb_T5Z2.root"
initpath = "../data/PhotoNeutronDMC_InitialTest10K_jswfix.mat"
savepath = "figs/bdt_all_scatters/"


def do_bdt(rq_var_names, rrq_var_names, new_var_info, calib_path, merge_path, init_path, det=14):

    variables, energies = data_loader(rq_var_names, rrq_var_names, new_var_info, calib_path, merge_path, init_path, det)

    # BDT time!
    rng = np.random.RandomState(1)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=200)
    
    matrix, targets, test_matrix, test_targets, test_dict = generate_fit_matrix(variables, rq_var_names +
                                                                                rrq_var_names +
                                                                                newvarinfo["names"],
                                                                                "Single?", 0.8, energies)
    bdt.fit(matrix, targets)
    
    print(bdt.feature_importances_)
    
    predictions = bdt.predict(test_matrix)
    scores = bdt.decision_function(test_matrix)
    num_tests = len(scores)
    
    split_scores = [[], []]
    for i in range(len(predictions)):
        split_scores[test_targets[i]].append(scores[i])
    
    return scores, split_scores, test_matrix, test_targets, test_dict, variables[0].keys()


if __name__ == '__main__':

    scores, split_scores, test_matrix, test_targets, test_dict, var_names = do_bdt(rqvarnames, rrqvarnames, newvarinfo,
                                                                                   calibpath, mergepath, initpath)
    print(var_names)
    plot_output(scores, split_scores, test_matrix, test_targets, test_dict, savepath)
