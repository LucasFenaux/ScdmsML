import numpy as np
from scipy.io import loadmat
import uproot
from .misc import cut_energy, generate_fit_matrix, generate_unsupervised_fit_matrix
import os
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from sklearn.decomposition import PCA


calib_paths = [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/calib_test_binary_01140301_0038_F_combined.root")]]
# [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/all_calib.root")]]
    # [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/calib_test_binary_01140301_0038_F_combined.root")]]
               # [True, os.path.relpath("../../data/V1_5_Photoneutron/combined/calib_test_binary_01140301_0038.root")]]
              # [True, os.path.relpath("../../data/V1_5_WIMP5/Processed/calib_test_binary_01150401_1725.root")],
               # [False, os.path.relpath("../../data/V1_5_CfVacuum/combined/calib_test_binary_01150401_1725.root")]]
# merge_paths = [[True, os.path.relpath("../../data/V1_5_WIMP5/Processed/merge_test_binary_01150401_1725.root")],
merge_paths = [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/test_binary_01140301_0038_F_combined.root")]]
    # [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/all_test.root")]]
    # [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/test_binary_01140301_0038_F_combined.root")]]
               # [True, os.path.relpath("../../data/V1_5_Photoneutron/combined/merge_test_binary_01140301_0038.root")]]
#               [True, os.path.relpath("../../data/V1_5_WIMP5/Processed/merge_test_binary_01150401_1725.root")],
               # [False, os.path.relpath("../../data/V1_5_CfVacuum/combined/merge_test_binary_01150401_1725.root")]]
# init_paths = [[True, os.path.relpath("../../data/V1_5_WIMP5/Input_SuperSim/input_5GeV_part2.mat")],
init_paths = [[True, os.path.relpath("../../data/V1_5_Photoneutron/combined/PhotoNeutronDMC_InitialTest10K_jswfix.mat")]]
              # [True, os.path.relpath("../../data/V1_5_Photoneutron/combined/PhotoNeutronDMC_InitialTest10K_jswfix.mat")]]
#              [True, os.path.relpath("../../data/V1_5_WIMP5/Input_SuperSim/input_5GeV_part2.mat")],
              # [False, os.path.relpath("../../data/V1_5_CfVacuum/Input_Supersim/Cf252_EStem_4.mat")]]
# dets = [4, 14, 4]
dets = [14]
#
# calib_paths = [[False, os.path.relpath("../../data/V1_5_CfVacuum/combined/calib_test_binary_01150401_1725.root")]]
# merge_paths = [[False, os.path.relpath("../../data/V1_5_CfVacuum/combined/merge_test_binary_01150401_1725.root")]]
# init_paths = [[False, os.path.relpath("../../data/V1_5_CfVacuum/Input_Supersim/Cf252_EStem_4.mat")]]
# dets = [4]


def sklearn_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, with_pca=0):
    train_data, train_targets, test_data, test_targets, test_dict, variables, feature_names = data_loader(rq_var_names,
                                                                                                          rrq_var_names,
                                                                                                          new_var_info,
                                                                                                          num_scatter_save_path)
    if with_pca > 0:
        pca = PCA(n_components=with_pca)
        train_data = pca.fit_transform(train_data)
        components = np.array(pca.components_)
        np.set_printoptions(suppress=True, precision=5)
        most_important_components = []
        for i in range(np.shape(components)[0]):
            most_important_comp = np.argmax(np.abs(components[i]))
            most_important_components.append(feature_names[most_important_comp])
            print(most_important_comp, feature_names[most_important_comp])
        print(most_important_components)
        test_data = pca.transform(test_data)
    return train_data, train_targets, test_data, test_targets, test_dict, variables, feature_names


def torch_data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, batch_size=256,
                      num_workers=1, pin_memory=False, with_pca=0):
    train_data, train_targets, test_data, test_targets, test_dict, variables, feature_names = data_loader(rq_var_names,
                                                                                                          rrq_var_names,
                                                                                                          new_var_info,
                                                                                                          num_scatter_save_path)
    if with_pca != 0:
        pca = PCA(n_components=with_pca)
        train_data = pca.fit_transform(train_data)
        components = np.array(pca.components_)
        np.set_printoptions(suppress=True, precision=5)
        most_important_components = []
        for i in range(np.shape(components)[0]):
            most_important_comp = np.argmax(np.abs(components[i]))
            most_important_components.append(feature_names[most_important_comp])
            print(most_important_comp, feature_names[most_important_comp])
        print(most_important_components)
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


def bg70V_sklearn_dataloader(rq_var_names, rrq_var_names, with_pca=0):
    calib_file_paths = [os.path.relpath("../../data/bg70V/calib_Prodv5-6-3_1505_bg70V_z14lite.root"),
                        os.path.relpath("../../data/bg70V/calib_Prodv5-6-3_1505r_bg70V_z14lite.root"),
                        os.path.relpath("../../data/bg70V/calib_Prodv5-6-3_1506_bg70V_z14lite.root")]
    merge_file_paths = [os.path.relpath("../../data/bg70V/merge_Prodv5-6-3_1505_bg70V_z14lite.root"),
                        os.path.relpath("../../data/bg70V/merge_Prodv5-6-3_1505r_bg70V_z14lite.root"),
                        os.path.relpath("../../data/bg70V/merge_Prodv5-6-3_1506_bg70V_z14lite.root")]
    file_dets = [14, 14, 14]

    train_data, test_data, test_dict, variables, feature_names = real_data_loader(rq_var_names, rrq_var_names)

    if with_pca > 0:
        pca = PCA(n_components=with_pca)
        train_data = pca.fit_transform(train_data)
        components = np.array(pca.components_)
        np.set_printoptions(suppress=True, precision=5)
        most_important_components = []
        for i in range(np.shape(components)[0]):
            most_important_comp = np.argmax(np.abs(components[i]))
            most_important_components.append(feature_names[most_important_comp])
            print(most_important_comp, feature_names[most_important_comp])
        print(most_important_components)
        test_data = pca.transform(test_data)

    return train_data, test_data, test_dict, variables, feature_names


def bg70_and_sim_sklearn_dataloader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path, with_pca=0):
    sim_train_data, train_targets, sim_test_data, test_targets, sim_test_dict, sim_variables, sim_feature_names = data_loader(rq_var_names,
                                                                                                          rrq_var_names,
                                                                                                          new_var_info,
                                                                                                          num_scatter_save_path)
    train_data, test_data, test_dict, variables, feature_names = real_data_loader(rq_var_names, rrq_var_names)
    all_data = np.ma.concatenate([np.array(sim_train_data), np.array(train_data)], axis=0)
    if with_pca > 0:
        pca = PCA(n_components=with_pca)
        pca = pca.fit(all_data)
        components = np.array(pca.components_)
        np.set_printoptions(suppress=True, precision=5)
        most_important_components = []
        for i in range(np.shape(components)[0]):
            most_important_comp = np.argmax(np.abs(components[i]))
            most_important_components.append(feature_names[most_important_comp])
            print(most_important_comp, feature_names[most_important_comp])
        print(most_important_components)
        test_data = pca.transform(test_data)
        sim_train_data = pca.transform(sim_train_data)
        train_data = pca.transform(train_data)
        sim_test_data = pca.transform(sim_test_data)

    return sim_train_data, train_targets, sim_test_data, test_targets, sim_test_dict, sim_variables, sim_feature_names, train_data, test_data, test_dict, variables, feature_names

def real_data_loader(rq_var_names, rrq_var_names):
    train_data = []
    test_data = []
    test_dict = []
    all_variables = []
    feature_names = []
    for file_idx in range(min(len(calib_paths), len(merge_paths), len(init_paths), len(dets))):
        calib_path = calib_paths[file_idx][1]
        merge_path = merge_paths[file_idx][1]
        init_path = init_paths[file_idx][1]
        det = dets[file_idx]
        calib = uproot.open(calib_path)["rrqDir"]["calibzip{}".format(det)]

        merge = uproot.open(merge_path)["rqDir"]
        variables = get_branches(merge, rq_var_names, det=det, tree=merge["zip{}".format(det)])
        variables = merge_variables(variables, get_branches(merge, rrq_var_names, tree=calib, det=det), rrq_var_names)
        energies = get_branches(merge, ["ptNF"], det=det, tree=calib, normalize=False)

        variables, energies = cut_energy(variables, energies, 20.)

        tr_data, t_data, t_dict, features = generate_unsupervised_fit_matrix(variables, rq_var_names +
                                                                                       rrq_var_names,
                                                                                       0.8, energies)
        train_data.extend(tr_data)
        test_data.extend(t_data)
        test_dict.extend(t_dict)
        all_variables.extend(variables)
        feature_names.extend(features)

    return train_data, test_data, test_dict, all_variables, feature_names


def data_loader(rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path):
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []
    test_dict = []
    all_variables = []
    feature_names = []

    # Loading in data from files
    for file_idx in range(min(len(calib_paths), len(merge_paths), len(init_paths), len(dets))):
        calib_path = calib_paths[file_idx][1]
        merge_path = merge_paths[file_idx][1]
        init_path = init_paths[file_idx][1]
        det = dets[file_idx]
        is_californium = calib_paths[file_idx][0]

        calib = uproot.open(calib_path)["rrqDir"]["calibzip{}".format(det)]

        merge = uproot.open(merge_path)["rqDir"]

        variables = get_branches(merge, rq_var_names, det=det, tree=merge["zip{}".format(det)])
        variables = merge_variables(variables, get_branches(merge, rrq_var_names, tree=calib, det=det), rrq_var_names)

        scatters, single_scatter = get_num_scatters(init_path, save_path=num_scatter_save_path, det=det)
        variables = add_to_variables(variables, "Single?", single_scatter, is_californium)

        energies = get_branches(merge, ["ptNF"], det=det, tree=calib, normalize=False)
        variables, energies = cut_energy(variables, energies, 20.)

        if len(new_var_info["names"]) != 0:
            for n in range(len(new_var_info["names"])):
                name = new_var_info["names"][n]
                in_var_names = new_var_info["inputs"][n]
                func = new_var_info["funcs"][n]

                in_vars = get_branches(merge, in_var_names, merge)
                variables = calculate_variable(variables, name, in_var_names, in_vars, func)

        tr_data, tr_targets, t_data, t_targets, t_dict, features = generate_fit_matrix(variables, rq_var_names +
                                                                                            rrq_var_names +
                                                                                            new_var_info["names"],
                                                                                            "Single?", 0.8, energies)
        train_data.extend(tr_data)
        train_targets.extend(tr_targets)
        test_data.extend(t_data)
        test_targets.extend(t_targets)
        test_dict.extend(t_dict)
        all_variables.extend(variables)
        feature_names.extend(features)

    return train_data, train_targets, test_data, test_targets, test_dict, all_variables, feature_names


def get_num_scatters(init_path, save_path, det=None, write=True):
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


def get_branches(merge, branches, det=None, tree=None, normalize=True):
    if tree is None:
        tree = merge["zip{}".format(det)]
    # print(tree.keys())
    # print(len(tree.keys()))
    evs_raw = merge["eventTree"]["EventNumber"].array().astype(int)
    raw_branches = {}
    for branch in branches:
        raw_branches[branch] = tree[branch].array()

    output = []

    max_vals = {}
    min_vals = {}

    for branch in branches:
        max_vals[branch] = 0.
        min_vals[branch] = 0.

    j = 0
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
            output[-1][branch] = raw_branches[branch][j]
            if output[-1][branch] > max_vals[branch]:
                max_vals[branch] = output[-1][branch]
            if output[-1][branch] < min_vals[branch]:
                min_vals[branch] = output[-1][branch]
        j += 1

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


def add_to_variables(variables, name, to_add, is_californium):
    num_var = len(variables)
    i = 0
    for key, value in to_add.items():
        if i >= num_var:
            break
        if is_californium:
            variables[i][name] = to_add[variables[i]["EV"]]
        else:
            variables[i][name] = value
        i += 1
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
