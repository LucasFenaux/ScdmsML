from math import cos, sin, radians
from ScdmsML.src.utils import data_loader
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
rq_var_names = ["PTNFchisq"]
rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")


def do_k_clustering(k=2, pca=0):
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)
    if pca != 0:
        train_data = PCA(n_components=pca).fit_transform(train_data)
    k_means = KMeans(n_init=100, max_iter=100, n_clusters=k, verbose=1).fit(train_data)
    print("targets proportions:0:", len(train_targets) - sum(train_targets), " | 1:", sum(train_targets))
    print("cluster centers:", k_means.cluster_centers_)
    print("labels: ", k_means.labels_)
    print("cluster proportions:")
    for cluster in np.unique(k_means.labels_):
        print(cluster, list(k_means.labels_).count(cluster))
    print("### Other metrics ###")
    t0 = time()
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

    visualize_k_clustering(train_data, train_targets, k_means, dims=pca, k=k)


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
        ax.text3D(reduced_data[targets == label, 1].mean(),
                  reduced_data[targets == label, 0].mean(),
                  reduced_data[targets == label, 2].mean() + 2, name,
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
    # plt.imshow(reduced_data, c=Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')
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
    do_k_clustering(k=2, pca=3)
