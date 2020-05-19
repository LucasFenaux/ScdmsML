import matplotlib.pyplot as plt
import numpy as np


def cut_energy(variables, energies, cut=10):
    new_vars = []
    new_energies = []
    for v in range(len(variables)):
        entry = variables[v]
        energy = energies[v]
        if energy["ptNF"] < cut:
            new_vars.append(entry)
            new_energies.append(energy)
    return new_vars, new_energies


def generate_fit_matrix(variables, feature_names, target_name, fraction, energies):
    matrix = []
    targets = []

    test_matrix = []
    test_targets = []
    test_dict = []

    for v in range(len(variables)):
        entry = variables[v]
        energy = energies[v]["ptNF"]
        sample = []
        sample_dict = {"ptNF": energy}
        for featurename in feature_names:
            sample.append(entry[featurename])
            sample_dict[featurename] = entry[featurename]
        if len(matrix) < fraction * len(variables):
            matrix.append(sample)
            targets.append(entry[target_name])
        else:
            test_matrix.append(sample)
            test_dict.append(sample_dict)
            test_targets.append(entry[target_name])
    return matrix, targets, test_matrix, test_targets, test_dict


def plot_output(scores, split_scores, test_matrix, test_targets, test_dict, savepath):
    plt.hist(split_scores, 50, histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single"],
             range=(-1, 0.3))
    plt.legend(loc=2)
    plt.xlabel("BDT Decision Function")
    plt.savefig(savepath + "scores.png")
    plt.show()

    t_single = []
    t_multi = []
    f_single = []
    f_multi = []
    purity = []

    cuts = np.linspace(-1, 1, num=1000)

    best_cut = 0
    for cut in cuts:
        t_single.append(0)
        t_multi.append(0)
        f_single.append(0)
        f_multi.append(0)
        purity.append(0)

        for i in range(len(split_scores[0])):
            if split_scores[0][i] < cut:
                t_multi[-1] += 1
            else:
                f_single[-1] += 1
        for j in range(len(split_scores[1])):
            if split_scores[1][j] < cut:
                f_multi[-1] += 1
            else:
                t_single[-1] += 1

        single_to_t = t_single[-1] + f_multi[-1]
        multi_to_t = t_multi[-1] + f_single[-1]
        if single_to_t == 0:
            t_single[-1] = 1.
            f_single[-1] = 1.
        elif multi_to_t == 0:
            t_single[-1] = 0.
            f_single[-1] = 0.
        else:
            t_single[-1] = 1. * t_single[-1] / single_to_t
            f_single[-1] = 1. * f_single[-1] / multi_to_t

    plt.clf()
    plt.plot(f_single, t_single)
    line = np.linspace(0, 1, 50)
    roc_sum = 0
    for i in range(len(f_single) - 1):
        dx = f_single[i] - f_single[i + 1]
        roc_sum += dx * (t_single[i] - f_single[i])

    print(roc_sum)

    plt.plot(line, line, "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(savepath + "roc.png")
    plt.show()
    plt.clf()

    old_spectrum = [[], [], []]
    new_spectrum = [[], [], []]
    for i in range(len(scores)):
        score = scores[i]
        old_spectrum[test_targets[i]].append(test_dict[i]["ptNF"])
        old_spectrum[2].append(test_dict[i]["ptNF"])
        if score > -0.1:
            new_spectrum[test_targets[i]].append(test_dict[i]["ptNF"])
            new_spectrum[2].append(test_dict[i]["ptNF"])

    plt.hist(old_spectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False,
             label=["Multiple", "Single", "All"])
    plt.xlim(0, 4)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath + "old_spectrum.png")
    plt.show()
    plt.clf()
    plt.hist(new_spectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                                "All"])
    plt.xlim(0, 4)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath + "new_spectrum.png")
    plt.show()

    plt.hist(old_spectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                                "All"])
    plt.xlim(0, 4)
    plt.ylim(0, 50)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath + "old_spectrum_zoomed.png")
    plt.show()
    plt.clf()
    plt.hist(new_spectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                                "All"])
    plt.xlim(0, 4)
    plt.ylim(0, 50)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath + "new_spectrum_zoomed.png")
    plt.show()