import numpy as np
from scipy.io import loadmat
import uproot
from .misc import cut_energy, generate_fit_matrix
import os
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from sklearn.decomposition import PCA


calib_path = os.path.relpath("../../data/calib_LibSimProdv5-4_pn_Sb_T5Z2.root")
merge_path = os.path.relpath("../../data/merge_LibSimProdv5-4_pn_Sb_T5Z2.root")
init_path = os.path.relpath("../../data/PhotoNeutronDMC_InitialTest10K_jswfix.mat")


def data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, det=14):
    # Loading in data from files
    calib = uproot.open(calib_path)["rrqDir"]["calibzip{}".format(det)]

    merge = uproot.open(merge_path)["rqDir"]["zip{}".format(det)]

    variables = get_branches(merge, rq_var_names, merge)
    variables = merge_variables(variables, get_branches(merge, rrq_var_names, calib), rrq_var_names)

    scatters, single_scatter = get_num_scatters(init_path, save_path=num_scatter_save_path)
    variables = add_to_variables(variables, "Single?", single_scatter)

    energies = get_branches(merge, ["ptNF"], calib, normalize=False)
    variables, energies = cut_energy(variables, energies, 20.)

    if len(new_var_info["names"]) != 0:
        for n in range(len(new_var_info["names"])):
            name = new_var_info["names"][n]
            in_var_names = new_var_info["inputs"][n]
            func = new_var_info["funcs"][n]

            in_vars = get_branches(merge, in_var_names, merge)
            variables = calculate_variable(variables, name, in_var_names, in_vars, func)

    train_data, train_targets, test_data, test_targets, test_dict = generate_fit_matrix(variables, rq_var_names +
                                                                                        rrq_var_names +
                                                                                        new_var_info["names"],
                                                                                        "Single?", 0.8, energies)

    return train_data, train_targets, test_data, test_targets, test_dict, variables


def torch_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, det=14, batch_size=256,
                      num_workers=1, pin_memory=False, with_pca=0):
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path,
                                                                                           det=14)
    if with_pca != 0:
        pca = PCA(n_components=with_pca)
        train_data = pca.fit_transform(train_data)
        components = np.array(pca.components_)
        np.set_printoptions(suppress=True, precision=5)
        for i in range(np.shape(components)[0]):
            print(np.argmax(np.abs(components[i])))
        print(components)
        test_data = pca.transform(test_data)

    train_data = torch.Tensor(train_data)
    train_targets = torch.Tensor(train_targets)
    train_targets = torch.nn.functional.one_hot(train_targets.to(torch.int64))

    test_data = torch.Tensor(test_data)
    test_targets = torch.Tensor(test_targets)
    test_targets = torch.nn.functional.one_hot(test_targets.to(torch.int64))

    train_dataset = TensorDataset(train_data, train_targets)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = TensorDataset(test_data, test_targets)
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def get_num_scatters(init_path, save_path, det=14, write=True):
    init = loadmat(init_path)
    # Get event number for each scatter
    ev_of_scatter = init["EV"][:, 0] + 1  # +1 here because starts at 0
    # Get detector for each scatter
    det_of_scatter = init["DT"][:, 0]
    # Get energy for each scatter
    energy_of_scatter = init["D3"][:, 0]
    # Energies of each scatter, to remove very weak scatters
    scatter_energies = {}

    last_idx = 0
    initial_ev = ev_of_scatter[last_idx]

    while last_idx < len(ev_of_scatter) - 1:
        # found boolean used so the entire list is not read each time
        found = False
        # Initialize new dict entry when current ev is greater than current maxscatterev
        max_scatter_ev = 0
        # Start at lastindx so only necessary indices are read
        for i in range(last_idx, len(ev_of_scatter)):
            # if evofscatter[i] >= 98000:
            # print(evofscatter[i])
            if ev_of_scatter[i] == initial_ev:
                if initial_ev > max_scatter_ev:
                    scatter_energies[initial_ev] = []
                    max_scatter_ev = initial_ev
                # If event of scatter matches event number AND detector matches,
                # increment scatter number by one
                if det_of_scatter[i] == det:
                    scatter_energies[initial_ev].append(energy_of_scatter[i])
                found = True
                # If it's the last scatter, set last_idx so the while loop ends
                if i == len(ev_of_scatter) - 1:
                    last_idx = i
            elif found:
                last_idx = i
                initial_ev = ev_of_scatter[i]
                break

    num_scatters = {}  # Output dict
    single_scatters = {}  # 1 if single scatter, 0 if multiple
    for ev in scatter_energies:
        num_scatters[ev] = 0
        max_energy = max(scatter_energies[ev])
        for energy in scatter_energies[ev]:
            if energy > 0.01 * max_energy:
                num_scatters[ev] += 1
        single_scatters[ev] = int(num_scatters[ev] == 1)

    # Write to file if requested. True by default
    if write:
        scatter_out = open(save_path, "w")
        scatter_out.write("EventNumber NumberOfScatters\n")
        for ev in num_scatters:
            scatter_out.write("{} {}\n".format(ev, num_scatters[ev]))
        scatter_out.close()

    return num_scatters, single_scatters


def get_branches(merge, branches, tree=None, normalize=True):
    if tree is None:
        tree = merge
    # print(tree.keys())
    # print(len(tree.keys()))
    evs_raw = merge["PTSIMEventNumber"].array().astype(int)
    raw_branches = {}
    for branch in branches:
        raw_branches[branch] = tree[branch].array()

    output = []

    max_vals = {}
    min_vals = {}

    for branch in branches:
        max_vals[branch] = 0.
        min_vals[branch] = 0.

    for i in range(len(evs_raw)):
        ev = evs_raw[i]
        # Some repeats in event number; continue if repeat
        if ev <= len(output):
            continue
        # Add new entry output
        output.append({})
        output[-1]["EV"] = ev
        # Add desired variables to output
        for branch in branches:
            output[-1][branch] = raw_branches[branch][i]
            if output[-1][branch] > max_vals[branch]:
                max_vals[branch] = output[-1][branch]
            if output[-1][branch] < min_vals[branch]:
                min_vals[branch] = output[-1][branch]

    if normalize:
        for v in range(len(output)):
            for branch in branches:
                if output[v][branch] == min_vals[branch]:
                    output[v][branch] = -9
                else:
                    output[v][branch] = np.log((output[v][branch] +
                                                np.abs(min_vals[branch])) / (
                                                           max_vals[branch] + np.abs(min_vals[branch])))
    return output


def add_to_variables(variables, name, to_add):
    for i in range(len(variables)):
        variables[i][name] = to_add[variables[i]["EV"]]
    return variables


def merge_variables(var1, var2, names2):
    for i in range(len(var1)):
        for name in names2:
            var1[i][name] = var2[i][name]
    return var1


def calculate_variable(variables, name, in_var_names, in_vars, function):
    for i in range(len(variables)):
        eventdict = in_vars[i]
        invarlist = []
        for invarname in in_var_names:
            invarlist.append(eventdict[invarname])
        outvar = function(invarlist)
        variables[i][name] = outvar
    return variables
