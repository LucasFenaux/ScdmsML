import matplotlib.pyplot as plt
import torch.nn as nn
from os import path
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import logging
# logging.basicConfig(filename='./custom_tracker.log', level=logging.INFO)


class CustomTracker(object):
    """ tracker that requires a sequential sampler for the loader used"""
    def __init__(self, model, device, loader, idx):
        self.model = model
        self.device = device
        self.loader = loader
        self.idx = idx
        self.sample = None
        self.sample_target = None
        assert self.idx < self.loader.__len__()
        self.mark_sample()
        self.assessment()
        self.best_test_accuracy = 0

    def mark_sample(self):
        i = 0
        for idx, (inputs, target) in enumerate(self.loader):
            if ((idx+1)*self.loader.batch_size)+i > self.idx:
                arr_idx = self.idx - i
                self.sample = inputs[arr_idx]
                self.sample_target = target[arr_idx]
                break
            else:
                i += self.loader.batch_size
        logging.log(level=logging.INFO, msg="sample logged: "+str(self.sample))
        logging.log(level=logging.INFO, msg="with target: "+str(self.sample_target))

    def assessment(self):
        i = 0
        for idx, (inputs, target) in enumerate(self.loader):
            # for i in range(inputs.size()[0]):
            #     if torch.all(torch.eq(inputs[i], self.sample)):
            if ((idx+1)*self.loader.batch_size)+i > self.idx:
                sample_idx = self.idx - i
                logging.log(level=logging.INFO, msg="found sample")
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                # logging.log(level=logging.INFO, msg="input assessment: " + str(inputs[sample_idx]))
                # logging.log(level=logging.INFO, msg="target assessment: " + str(target[sample_idx]))
                logging.log(level=logging.INFO, msg="output assessment: " + str(output[sample_idx]))
                break
            else:
                i += self.loader.batch_size

    def update_best_accuracy(self, accuracy):
        if accuracy > self.best_test_accuracy:
            self.best_test_accuracy = accuracy

    def get_best_accuracy(self):
        return self.best_test_accuracy


class NDMinMaxScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
            print("orig_shape:", self._orig_shape)
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


def get_tensorboard_log_dir():
    if not path.exists('../results/files/tb_logs'):
        return path.join("../results/files/tb_logs")
    i = 1
    while path.exists('../results/files/tb_logs_{}'.format(i)):
        i += 1
    return path.join("../results/files/tb_logs_{}".format(i))


def cut_energy(variables, energies, cut=10):
    """ Cut events based on the given cut variable value"""
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
    """" Generates the data matrices as well as the target arrays"""
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
    return matrix, targets, test_matrix, test_targets, test_dict, feature_names


def generate_unsupervised_fit_matrix(variables, feature_names, fraction, energies):
    """ Generates the data matrix but no target array"""
    matrix = []

    test_matrix = []
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
        else:
            test_matrix.append(sample)
            test_dict.append(sample_dict)

    return matrix, test_matrix, test_dict, feature_names


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


class ListModule(nn.Module):
    """ Module used for Pytorch to construct module made of multiple layers """
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
