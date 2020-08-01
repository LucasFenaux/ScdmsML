from math import cos, sin, radians
from ScdmsML.src.utils import sklearn_data_loader, bg70V_sklearn_dataloader, bg70_and_sim_sklearn_dataloader
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import numpy as np
from time import time

# Global Variable
det = 14
# All available RQs and RRQs
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
                'PDbias', 'PDbs', 'PDbspost', 'PDgain', 'PDnorm', 'PDsat', 'PDstd', 'PS1INTall', 'PS1OFamps',
                'PS1OFamps0', 'PS1OFchisq', 'PS1OFchisqLF', 'PS1OFdelay', 'PS2INTall', 'PS2OFamps', 'PS2OFamps0',
                'PS2OFchisq', 'PS2OFchisqLF', 'PS2OFdelay', 'PTNFamps', 'PTNFamps0', 'PTNFchisq', 'PTNFdelay',
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
# rrq_var_names = ['paOF', 'paOF0', 'paOF0c', 'paOFc', 'paampres', 'padelayres', 'pbOF', 'pbOF0', 'pbOF0c', 'pbOFc',
#                  'pbampres', 'pbdelayres', 'pcOF', 'pcOF0', 'pcOF0c', 'pcOFc', 'pcampres', 'pcdelayres', 'pdOF',
#                  'pdOF0', 'pdOF0c', 'pdOFc', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
#                  'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
#                  'pprimechanOFWK', 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prxypartOF', 'psumOF',
#                  'psumiOF', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0c', 'ptNF0uc', 'ptNFc', 'ptNFuc', 'ptOF', 'ptOF0',
#                  'ptOF0c', 'ptOF0uc', 'ptOFc', 'ptOFuc', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qiOF0',
#                  'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']
rrq_var_names = []
# rq_var_names = ['PATFPeflag', 'PBTFPchisq', 'PTPSDint0to1', 'PCWKr10', 'QSOFdiscreteDelay', 'PTPSDint0to1', 'PTNFamps0',
#                 'PCOFdelay', 'QOOFdiscreteVolts', 'QIOFvolts', 'PTlfnoise1OFdelay', 'PTglitch1OFdelay', 'PTOFdelay',
#                 'PDOFamps0', 'PTPSDint0to1', 'PTPSDint1to10', 'PDINTall', 'QSOFdelay', 'PTglitch1OFamps0', 'QIOFvolts0',
#                 'QOOFvolts0']

# rq_var_names = []
# rrq_var_names = ['paOF', 'paampres', 'padelayres', 'pbOF',
#                  'pbampres', 'pbdelayres', 'pcOF', 'pcampres', 'pcdelayres', 'pdOF',
#                  'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
#                  'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
#                  'pprimechaniOF', 'prdelWK', 'ptNF',
#                  'ptOF', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qoOF', 'qrpartOF']


# yet to test
# rq_var_names = ['PCTFPchisq', 'PTOFamps']
# rrq_var_names = ['ptNF0c', 'pminrtOFWK_1070']


# RQs and RRQs from pca
# Similar shape RQs and RRQs

# Group 1: best NN acc: 0.6322

# Results:
# [[12201.     0.]
#  [ 7098.     0.]]
# Final Torch Loss:  0.23185645601422739
# Final Torch Err:  tensor(0.6322)
# rq_var_names = ["PATFPeflag"]
# rrq_var_names = ['pminrtWK_1070']

# Results:
# [[12201.     0.]
#  [ 7098.     0.]]
# Final Torch Loss:  0.23185370656513843
# Final Torch Err:  tensor(0.6322)
# rq_var_names = ['PBTFPeflag']
# rrq_var_names = ['pminrtWK_1040']

# Results:
# [[11896.   305.]
#  [ 6374.   724.]]
# Final Torch Loss:  0.22629748456169252
# Final Torch Err:  tensor(0.6539)
# rq_var_names = ['PCTFPeflag']
# rrq_var_names = ['ptNF0uc']

# Results:
# [[12201.     0.]
#  [ 7098.     0.]]
# Final Torch Loss:  0.23185044745193642
# Final Torch Err:  tensor(0.6322)
# rq_var_names = ['PDTFPeflag']
# rrq_var_names = ['pminrtWK_10100']

# Group 2: best NN acc: 0.6539

# Results:
# [[11896.   305.]
#  [ 6374.   724.]]
# Final Torch Loss:  0.22624048697243063
# Final Torch Err:  tensor(0.6539)
# rq_var_names = ["PBTFPoffset"]
# rrq_var_names = ["ptNF"]

# Results:
# [[11896.   305.]
#  [ 6374.   724.]]
# Final Torch Loss:  0.22619099729476538
# Final Torch Err:  tensor(0.6539)
# rq_var_names = ["PBTFPchisq"]
# rrq_var_names = ["ptNF0"]

# Group 3: best NN acc: 0.6354
# Results:
# [[11748.   453.]
#  [ 6583.   515.]]
# Final Torch Loss:  0.23117222767801884
# Final Torch Err:  tensor(0.6354)
# rq_var_names = ["PTPSDint0to1"]
# rrq_var_names = ['pminrtOFWK_10100']

new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")


def do_k_clustering(k=2, pca=0):
    # sim_train_data, sim_train_targets, sim_test_data, sim_test_targets, sim_test_dict, sim_variables, sim_feature_names\
    #     = sklearn_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, with_pca=pca)
    # train_data, test_data, test_dict, variables, feature_names = bg70V_sklearn_dataloader(rq_var_names, rrq_var_names, with_pca=pca)
    sim_train_data, sim_train_targets, sim_test_data, sim_test_targets, sim_test_dict, sim_variables, sim_feature_names,\
    train_data, test_data, test_dict, variables, feature_names = bg70_and_sim_sklearn_dataloader(rq_var_names,
                                                                                                 rrq_var_names,
                                                                                                 new_var_info,
                                                                                                 num_scatter_save_path,
                                                                                                 with_pca=pca)
    all_data = np.ma.concatenate([sim_train_data, train_data], axis=0)
    k_means = KMeans(n_init=100, max_iter=100, n_clusters=k, verbose=0).fit(all_data)
    # print("targets proportions:0:", len(train_targets) - sum(train_targets), " | 1:", sum(train_targets))
    # print("cluster centers:", k_means.cluster_centers_)
    # print("labels: ", k_means.labels_)
    # k_means = k_means.fit(sim_train_data)
    print("cluster proportions:")
    for cluster in np.unique(k_means.labels_):
        print(cluster, list(k_means.labels_).count(cluster))
    print("### Other metrics ###")
    t0 = time()

    sim_preds = k_means.predict(sim_test_data)
    print(sim_preds)
    print("sim target proportions:0:", len(sim_test_targets) - sum(sim_test_targets), " | 1:", sum(sim_test_targets))
    print("predicted proportions:0:", len(sim_preds) - sum(sim_preds), " | 1:", sum(sim_preds))
    # print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
    #       % ("k_means", (time() - t0), k_means.inertia_,
    #          metrics.homogeneity_score(train_targets, k_means.labels_),
    #          metrics.completeness_score(train_targets, k_means.labels_),
    #          metrics.v_measure_score(train_targets, k_means.labels_),
    #          metrics.adjusted_rand_score(train_targets, k_means.labels_),
    #          metrics.adjusted_mutual_info_score(train_targets,  k_means.labels_),
    #          metrics.silhouette_score(train_data, k_means.labels_,
    #                                   metric='euclidean',
    #                                   sample_size=len(train_targets))))

    visualize_k_clustering(sim_test_data, sim_test_targets, k_means, dims=pca, k=k)


def visualize_k_clustering(reduced_data, targets, kmeans, dims, k):
    if dims == 2:
        visualize_2d(reduced_data, targets, kmeans)
    elif dims == 3:
        visualize_3d(reduced_data, targets, kmeans, k)
    else:
        print("Dimension of the data not fit for visual representation")


def visualize_3d(reduced_data, targets, kmeans, k):
    name, est = ('k_mean', kmeans)

    fignum = 1
    title = "Predicted clusters"

    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig)
    labels = est.labels_
    xs = reduced_data[:, 0]
    ys = reduced_data[:, 1]
    zs = reduced_data[:, 2]

    ax.scatter(xs, ys, zs, c=labels, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(title)
    ax.dist = 12
    fignum = fignum + 1

    fig.savefig("../results/figs/3D_clusters_predicted.png")

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig)

    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(reduced_data[:, 1].mean(),
                  reduced_data[:, 0].mean(),
                  reduced_data[:, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(targets, [1, 0, 2]).astype(np.float)
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title('Ground Truth')
    ax.dist = 12

    fig.show()
    fig.savefig("../results/figs/3D_clusters_ground_truth.png")


def visualize_2d(reduced_data, targets, kmeans):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('Predicted clusters. Centroids marked as white X')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("../results/figs/2D_clusters_predicted.png")
    plt.show()

    # Plot Ground Truth
    plt.figure(2)
    plt.clf()
    Z = np.array(targets)
    y = np.choose(Z, [1, 0, 2]).astype(np.float)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, edgecolors="k")
    plt.title("Ground Truth")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("../results/figs/2D_clusters_ground_truth.png")
    plt.show()


if __name__ == '__main__':
    do_k_clustering(k=2, pca=2)

