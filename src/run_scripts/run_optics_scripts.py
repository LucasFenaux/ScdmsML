from math import cos, sin, radians
from ScdmsML.src.utils import bg70_and_sim_sklearn_dataloader, compute_accuracy
from sklearn.cluster import KMeans, OPTICS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ScdmsML.src.utils import build_confusion_matrix, sklearn_data_loader
from sklearn.neural_network import MLPClassifier
import torch


import os
import numpy as np
from time import time


# All available RQs and RRQs
# rq_var_names = ['PAOFamps', 'PAOFamps0', 'PAOFchisq', 'PAOFchisqLF', 'PAOFdelay', 'PATFPamp', 'PATFPchisq',
#                  'PATFPeflag', 'PATFPint', 'PATFPoffset', 'PATFPtau', 'PAWKf20', 'PAWKf40', 'PAWKf80', 'PAWKf90',
#                  'PAWKf95', 'PAWKmax', 'PAWKr10', 'PAWKr100', 'PAWKr20', 'PAWKr30', 'PAWKr40', 'PAWKr50', 'PAWKr60',
#                  'PAWKr70', 'PAWKr80', 'PAWKr90', 'PAWKr95', 'PAbias', 'PAbs', 'PAbspost', 'PAgain', 'PAnorm', 'PAsat',
#                  'PAstd', 'PBINTall', 'PBOFamps', 'PBOFamps0', 'PBOFchisq', 'PBOFchisqLF', 'PBOFdelay', 'PBTFPamp',
#                  'PBTFPchisq', 'PBTFPeflag', 'PBTFPint', 'PBTFPoffset', 'PBTFPtau', 'PBWKf20', 'PBWKf40', 'PBWKf80',
#                  'PBWKf90', 'PBWKf95', 'PBWKmax', 'PBWKr10', 'PBWKr100', 'PBWKr20', 'PBWKr30', 'PBWKr40', 'PBWKr50',
#                  'PBWKr60', 'PBWKr70', 'PBWKr80', 'PBWKr90', 'PBWKr95', 'PBbias', 'PBbs', 'PBbspost', 'PBgain',
#                 'PBnorm', 'PBsat', 'PBstd', 'PCINTall', 'PCOFamps', 'PCOFamps0', 'PCOFchisq', 'PCOFchisqLF',
#                 'PCOFdelay', 'PCTFPamp', 'PCTFPchisq', 'PCTFPeflag', 'PCTFPint', 'PCTFPoffset', 'PCTFPtau', 'PCWKf20',
#                 'PCWKf40', 'PCWKf80', 'PCWKf90', 'PCWKf95', 'PCWKmax', 'PCWKr10', 'PCWKr100', 'PCWKr20', 'PCWKr30',
#                 'PCWKr40', 'PCWKr50', 'PCWKr60', 'PCWKr70', 'PCWKr80', 'PCWKr90', 'PCWKr95', 'PCbias', 'PCbs',
#                 'PCbspost', 'PCgain', 'PCnorm', 'PCsat', 'PCstd', 'PDINTall', 'PDOFamps', 'PDOFamps0', 'PDOFchisq',
#                 'PDOFchisqLF', 'PDOFdelay', 'PDTFPamp', 'PDTFPchisq', 'PDTFPeflag', 'PDTFPint', 'PDTFPoffset',
#                 'PDTFPtau', 'PDWKf20', 'PDWKf40', 'PDWKf80', 'PDWKf90', 'PDWKf95', 'PDWKmax', 'PDWKr10', 'PDWKr100',
#                 'PDWKr20', 'PDWKr30', 'PDWKr40', 'PDWKr50', 'PDWKr60', 'PDWKr70', 'PDWKr80', 'PDWKr90', 'PDWKr95',
#                 'PDbias', 'PDbs', 'PDbspost', 'PDgain', 'PDnorm', 'PDsat', 'PDstd', 'PS1INTall', 'PS1OFamps',
#                 'PS1OFamps0', 'PS1OFchisq', 'PS1OFchisqLF', 'PS1OFdelay', 'PS2INTall', 'PS2OFamps', 'PS2OFamps0',
#                 'PS2OFchisq', 'PS2OFchisqLF', 'PS2OFdelay', 'PTNFamps', 'PTNFamps0', 'PTNFchisq', 'PTNFdelay',
#                 'PTOFamps', 'PTOFamps0', 'PTOFchisq', 'PTOFchisqLF', 'PTOFdelay', 'PTPSDint0to1', 'PTPSDint1to10',
#                 'PTPSDintall', 'PTglitch1OFamps', 'PTglitch1OFamps0', 'PTglitch1OFchisq', 'PTglitch1OFchisqLF',
#                 'PTglitch1OFdelay', 'PTlfnoise1OFamps', 'PTlfnoise1OFamps0', 'PTlfnoise1OFchisq',
#                 'PTlfnoise1OFchisqLF', 'PTlfnoise1OFdelay', 'QIF5base', 'QIF5chisq', 'QIF5volts', 'QIOFchisq',
#                 'QIOFchisqBase', 'QIOFdelay', 'QIOFdiscreteChisq', 'QIOFdiscreteDelay', 'QIOFdiscreteVolts', 'QIOFflag',
#                 'QIOFvolts', 'QIOFvolts0', 'QIbias', 'QIbiastime', 'QIbs', 'QIbspost', 'QIgain', 'QInorm', 'QIsat',
#                 'QIstd', 'QOF5base', 'QOF5chisq', 'QOF5volts', 'QOOFchisq', 'QOOFchisqBase', 'QOOFdelay',
#                 'QOOFdiscreteChisq', 'QOOFdiscreteDelay', 'QOOFdiscreteVolts', 'QOOFflag', 'QOOFvolts', 'QOOFvolts0',
#                 'QObias', 'QObiastime', 'QObs', 'QObspost', 'QOgain', 'QOnorm', 'QOsat', 'QOstd', 'QSF5satdelay',
#                 'QSOFchisq', 'QSOFchisqBase', 'QSOFdelay', 'QSOFdiscreteChisq', 'QSOFdiscreteDelay']

# the ones available for both the simulated and the real data
rrq_var_names = ['paOF', 'paOF0', 'paOF0c', 'paOFc', 'paampres', 'padelayres', 'pbOF', 'pbOF0', 'pbOF0c', 'pbOFc',
                 'pbampres', 'pbdelayres', 'pcOF', 'pcOF0', 'pcOF0c', 'pcOFc', 'pcampres', 'pcdelayres', 'pdOF',
                 'pdOF0', 'pdOF0c', 'pdOFc', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
                 'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
                 'pprimechanOFWK', 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prxypartOF', 'psumOF',
                 'psumiOF', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0c', 'ptNF0uc', 'ptNFc', 'ptNFuc', 'ptOF', 'ptOF0',
                 'ptOF0c', 'ptOF0uc', 'ptOFc', 'ptOFuc', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qiOF0',
                 'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']

rq_var_names = ['PAOFamps', 'PAOFamps0', 'PAOFchisq', 'PAOFchisqLF', 'PAOFdelay', 'PATFPamp', 'PATFPchisq',
                 'PATFPeflag', 'PATFPint', 'PATFPoffset', 'PATFPtau', 'PAWKf20', 'PAWKf40', 'PAWKf80', 'PAWKf90',
                 'PAWKf95', 'PAWKmax', 'PAWKr10', 'PAWKr100', 'PAWKr20', 'PAWKr30', 'PAWKr40', 'PAWKr50', 'PAWKr60',
                 'PAWKr70', 'PAWKr80', 'PAWKr90', 'PAWKr95', 'PAbias', 'PAbs', 'PAbspost', 'PAgain', 'PAnorm', 'PAsat',
                 'PAstd', 'PBINTall', 'PBOFamps', 'PBOFamps0', 'PBOFchisq', 'PBOFchisqLF', 'PBOFdelay', 'PBTFPamp',
                 'PBTFPchisq', 'PBTFPeflag', 'PBTFPint', 'PBTFPoffset', 'PBTFPtau', 'PBWKf20', 'PBWKf40', 'PBWKf80',
                 'PBWKf90', 'PBWKf95', 'PBWKmax', 'PBWKr10', 'PBWKr100', 'PBWKr20', 'PBWKr30', 'PBWKr40', 'PBWKr50',
                 'PBWKr60', 'PBWKr70', 'PBWKr80', 'PBWKr90', 'PBWKr95', 'PBbias', 'PBbs', 'PBbspost', 'PBgain',
                'PBnorm', 'PBsat', 'PBstd', 'PCINTall', 'PCOFamps', 'PCOFamps0', 'PCOFchisq', 'PCOFchisqLF',
                'PCOFdelay', 'PCTFPamp', 'PCTFPchisq', 'PCTFPeflag', 'PCTFPint', 'PCTFPoffset', 'PCTFPtau', 'PCWKf20',
                'PCWKf40', 'PCWKf80', 'PCWKf90', 'PCWKf95', 'PCWKmax', 'PCWKr10', 'PCWKr100', 'PCWKr20', 'PCWKr30',
                'PCWKr40', 'PCWKr50', 'PCWKr60', 'PCWKr70', 'PCWKr80', 'PCWKr90', 'PCWKr95', 'PCbias', 'PCbs',
                'PCbspost', 'PCgain', 'PCnorm', 'PCsat', 'PCstd', 'PDINTall', 'PDOFamps', 'PDOFamps0', 'PDOFchisq',
                'PDOFchisqLF', 'PDOFdelay', 'PDTFPamp', 'PDTFPchisq', 'PDTFPeflag', 'PDTFPint', 'PDTFPoffset',
                'PDTFPtau', 'PDWKf20', 'PDWKf40', 'PDWKf80', 'PDWKf90', 'PDWKf95', 'PDWKmax', 'PDWKr10', 'PDWKr100',
                'PDWKr20', 'PDWKr30', 'PDWKr40', 'PDWKr50', 'PDWKr60', 'PDWKr70', 'PDWKr80', 'PDWKr90', 'PDWKr95',
                'PDbias', 'PDbs', 'PDbspost', 'PDgain', 'PDnorm', 'PDsat', 'PDstd'
                , 'PTNFamps', 'PTNFamps0', 'PTNFchisq', 'PTNFdelay',
                'PTOFamps', 'PTOFamps0', 'PTOFchisq', 'PTOFchisqLF', 'PTOFdelay', 'PTPSDint0to1', 'PTPSDint1to10',
                'PTPSDintall', 'PTglitch1OFamps', 'PTglitch1OFamps0', 'PTglitch1OFchisq', 'PTglitch1OFchisqLF',
                'PTglitch1OFdelay', 'PTlfnoise1OFamps', 'PTlfnoise1OFamps0', 'PTlfnoise1OFchisq',
                'PTlfnoise1OFchisqLF', 'PTlfnoise1OFdelay', 'QIF5base', 'QIF5chisq', 'QIF5volts', 'QIOFchisq',
                'QIOFchisqBase', 'QIOFdelay', 'QIOFdiscreteChisq', 'QIOFdiscreteDelay', 'QIOFdiscreteVolts', 'QIOFflag',
                'QIOFvolts', 'QIOFvolts0', 'QIbias', 'QIbiastime', 'QIbs', 'QIbspost', 'QIgain', 'QInorm', 'QIsat',
                'QIstd', 'QOF5base', 'QOF5chisq', 'QOF5volts', 'QOOFchisq', 'QOOFchisqBase', 'QOOFdelay',
                'QOOFdiscreteChisq', 'QOOFdiscreteDelay', 'QOOFdiscreteVolts', 'QOOFflag', 'QOOFvolts', 'QOOFvolts0',
                'QObias', 'QObiastime', 'QObs', 'QObspost', 'QOgain', 'QOnorm', 'QOsat', 'QOstd', 'QSF5satdelay',
                'QSOFchisq', 'QSOFchisqBase', 'QSOFdelay', 'QSOFdiscreteChisq', 'QSOFdiscreteDelay']
# rrq_var_names = []
# rq_var_names = []


new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 6
batch_size = 256


def do_optics_with_sim(pca=0, load=False):
    sim_train_data, sim_train_targets, sim_test_data, sim_test_targets, sim_test_dict, sim_variables, sim_feature_names\
        = sklearn_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, with_pca=pca)

    print(np.shape(sim_train_data))
    if not load:
        optics = OPTICS(min_samples=2, n_jobs=-1).fit(sim_train_data)

        # To see all the created clusters
        # -1 represents what the algorithm considers as noise, if there is too many elements, re-run with modified
        # min_samples and max_eps values

        print("number of clusters:", len(np.unique(optics.labels_)))
        print("cluster proportions:")
        for cluster in np.unique(optics.labels_):
            print(cluster, list(optics.labels_).count(cluster))
        print(optics.cluster_hierarchy_)

        # Optics does not create a fixed amount of clusters every run, so we look at each created cluster and relabel the
        # data depending on the majority of the elements in the cluster
        cluster_mapping = {}
        sim_train_targets = np.array(sim_train_targets)

        for cluster in np.unique(optics.labels_):
            if cluster == -1:
                continue
            indices = np.array([index for index, value in enumerate(optics.labels_) if value == cluster])
            cluster_labels = sim_train_targets[indices]
            if sum(cluster_labels) > float(len(cluster_labels)) / 2.:
                cluster_mapping[cluster] = 1
            else:
                cluster_mapping[cluster] = 0

        pred_targets = []
        for index, value in enumerate(optics.labels_):
            if value == -1:  # the point is considered as noise so we just label it as its original label
                pred_targets.append(sim_train_targets[index])
            else:
                pred_targets.append(cluster_mapping[value])
        pred_targets = np.array(pred_targets)

        # since the optics process can take a long time, once one has found good hyper-parameters, instead of re-running
        # it every time, they could just reload the saved targets
        np.save("recomputed_sim_targets.npy", pred_targets)
    else:
        pred_targets = np.load("recomputed_sim_targets.npy")

    model = MLPClassifier(hidden_layer_sizes=(100, 100), solver="sgd", activation="relu"
                          , max_iter=1000, n_iter_no_change=50, verbose=1).fit(sim_train_data, pred_targets)
    acc = model.score(sim_test_data, sim_test_targets)

    print("Sklearn acc:", acc)

    return


def do_optics_with_real_and_sim(pca=0, load=False):
    """We perform OPTICS clustering on both the simulated and the real photoneutron data, then relabel the simulated
    data and label the real data using the created clusters
    !! We do not relabel the testing data !!"""
    sim_train_data, sim_train_targets, sim_test_data, sim_test_targets, sim_test_dict, sim_variables, sim_feature_names,\
        train_data, test_data, test_dict, variables, feature_names = bg70_and_sim_sklearn_dataloader(rq_var_names,
                                                                                                     rrq_var_names,
                                                                                                     new_var_info,
                                                                                                     num_scatter_save_path,
                                                                                                     with_pca=pca)
    all_data = np.ma.concatenate([sim_train_data, train_data], axis=0)
    print(np.shape(all_data))

    # create fake targets for real data
    fake_targets = np.array([2]*np.shape(train_data)[0])
    assert np.shape(fake_targets)[0] == np.shape(train_data)[0]

    sim_train_targets = np.array(sim_train_targets)
    all_targets = np.hstack((sim_train_targets, fake_targets))
    assert np.shape(all_data)[0] == np.shape(all_targets)[0]

    if not load:

        optics = OPTICS(min_samples=2, max_eps=1, n_jobs=-1).fit(all_data)

        # visualize_k_clustering(sim_test_data, sim_test_targets, optics, dims=pca, k=k)
        print("number of clusters:", len(np.unique(optics.labels_)))
        print("cluster proportions:")
        for cluster in np.unique(optics.labels_):
            print(cluster, list(optics.labels_).count(cluster))
        print(optics.cluster_hierarchy_)

        cluster_mapping = {}

        for cluster in np.unique(optics.labels_):
            indices = np.array([index for index, value in enumerate(optics.labels_) if value == cluster])
            if cluster == -1:
                continue
            cluster_labels = all_targets[indices]
            cluster_score = 0
            cluster_points = 0
            for label in cluster_labels:
                if label == 2:  # real data so we can't count it in the relabelling process
                    continue
                else:
                    cluster_score += label
                    cluster_points += 1
            if cluster_points > 2*cluster_score:
                cluster_mapping[cluster] = 0
            else:
                cluster_mapping[cluster] = 1

        pred_targets = []
        for index, value in enumerate(optics.labels_):
            if value == -1:  # the point is considered as noise so we just label it as its original label
                if index >= np.shape(sim_train_targets)[0]:  # it's real data
                    pred_targets.append(2)
                else:
                    pred_targets.append(sim_train_targets[index])
            else:
                pred_targets.append(cluster_mapping[value])
        pred_targets = np.array(pred_targets)
        np.save("recomputed_real_and_sim_targets.npy", pred_targets)

    else:
        pred_targets = np.load("recomputed_real_and_sim_targets.npy")

    # we ignore the noisy real data
    indices = np.where(pred_targets == 2)
    all_data = np.delete(all_data, indices, axis=0)
    pred_targets = np.delete(pred_targets, indices)

    model = MLPClassifier(hidden_layer_sizes=(100, 100), solver="sgd", activation="relu"
                          , max_iter=1000, n_iter_no_change=50, verbose=1).fit(all_data, pred_targets)
    acc = model.score(sim_test_data, sim_test_targets)

    print("Sklearn acc:", acc)
    

if __name__ == '__main__':
    # do_optics_with_sim(pca=0, load=False)
    do_optics_with_real_and_sim(pca=2, load=True)
