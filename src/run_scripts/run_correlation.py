import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from ScdmsML.src.utils import torch_data_loader, build_confusion_matrix, data_loader, sklearn_data_loader

from math import cos, sin, radians
import os

new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/nn_numscatters.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")
num_workers = 6
batch_size = 256
number_of_classes = 2

criterion = torch.nn.BCELoss()  # Change to use different loss function


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

rrq_var_names = ['paOF', 'paOF0', 'paOF0c', 'paOFc', 'paampres', 'padelayres', 'pbOF', 'pbOF0', 'pbOF0c', 'pbOFc',
                 'pbampres', 'pbdelayres', 'pcOF', 'pcOF0', 'pcOF0c', 'pcOFc', 'pcampres', 'pcdelayres', 'pdOF',
                 'pdOF0', 'pdOF0c', 'pdOFc', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
                 'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
                 'pprimechanOFWK', 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prxypartOF', 'psumOF',
                 'psumiOF', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0c', 'ptNF0uc', 'ptNFc', 'ptNFuc', 'ptOF', 'ptOF0',
                 'ptOF0c', 'ptOF0uc', 'ptOFc', 'ptOFuc', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qiOF0',
                 'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']

rrqs_to_check = ['paOF', 'paampres', 'padelayres', 'pbOF',
                 'pbampres', 'pbdelayres', 'pcOF', 'pcampres', 'pcdelayres', 'pdOF',
                 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
                 'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
                 'pprimechaniOF', 'prdelWK', 'ptNF',
                 'ptOF', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qoOF', 'qrpartOF']

def do_correlation():

    train_data, train_targets, test_data, test_targets, test_dict, variables, feature_names = sklearn_data_loader \
        (rq_var_names, rrq_var_names, new_var_info, num_scatter_save_path)

    data = pd.DataFrame(np.array(train_data, dtype=np.float))
    print(data.isnull().values.any())
    corr = data.corr()
    print(corr.isnull().values.any())
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) >= 0.8:
                if columns[j]:
                    columns[j] = True
    selected_columns = data.columns[columns]
    data = data[selected_columns]
    indexes = np.array(selected_columns.values)
    feature_names = [feature_names[j] for j in indexes]
    print(feature_names)
    rrqs_indices = []
    for rrq in rrqs_to_check:
        rrqs_indices.append(feature_names.index(rrq))
    print(rrqs_indices)
    corr = data.corr()

    # sns.heatmap(np.triu(corr), yticklabels=feature_names, xticklabels=feature_names, annot=True, fmt='.1g',
    #             vmin=-1, vmax=1, center=0, linewidths=1, linecolor='black', cbar=False)

    # Print out most correlated values
    c = corr.abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")

    wanted_corr = []
    for idx in rrqs_indices:
        idx_corr = []
        for i in rrqs_indices:
            idx_corr.append(so[idx][i])
        wanted_corr.append(idx_corr)
    print(wanted_corr)
    wanted_corr = np.array(wanted_corr)
    np.save('corr_data.npy', wanted_corr)
    heatmap = sns.heatmap(wanted_corr, yticklabels=rrqs_to_check, xticklabels=rrqs_to_check,
                vmin=-1, vmax=1, center=0, linewidths=1, linecolor='black', cmap='Blues')
    heatmap.figure.savefig('corr_heatmap.png')
    print('done')
    # print(np.argmax(so[46624]))
    # max_idx = 0
    # max = 0
    # for i in range(len(so)):
    #     if so[i] != np.NaN and so[i] > max:
    #         max = so[i]
    #         max_idx = i
    #         break
    # print(max_idx, max)
    # print(so[-10000:-3000]) #change index here for diff print


def do_saved_correlation():
    corr = np.load('corr_data.npy')
    print(corr)
    # heatmap = sns.heatmap(corr, yticklabels=rrqs_to_check, xticklabels=rrqs_to_check,
    #             vmin=-1, vmax=1, center=0, linewidths=1, linecolor='black', cmap='Blues')
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, center=0, yticklabels=rrqs_to_check, xticklabels=rrqs_to_check,
                          linecolor='black', linewidths=0.2, cmap='coolwarm')
    heatmap.figure.savefig('corr_heatmap.png')

if __name__ == '__main__':
    do_correlation()
    # do_saved_correlation()