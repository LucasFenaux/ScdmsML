from math import cos, sin, radians
import numpy as np
from ScdmsML.src.utils import data_loader
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import os

# Global Variable
det = 14
rq_var_names = ["PTNFchisq"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc"]
# rrq_var_names = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK", "pminrtWK_10100"
#                  , "pminrtWK_1040", "pminrtWK_1070", "pminrtOFWK_10100", "pminrtOFWK_1040", "pminrtOFWK_1070",
#                  "pprimechaniOF", "prdelWK", "psumOF", "psumiOF", "ptNF", "ptNF0", "ptNF0uc", "ptOF", "ptOF0",
#                  "ptOF1X2P", "ptOF1X2P0", "ptOF1X2P0uc", "ptOF1X2Puc", "ptOF1X2R", "ptOF1X2R0", "ptftWK_9520",
#                  "pxpartOF"]
# rrq_var_names = ['DetType', 'Empty', 'pa2tdelr', 'paOF', 'paOF0', 'paOF1X2P', 'paOF1X2P0', 'paOF1X2R', 'paOF1X2R0',
#                  'paampres', 'padelayres', 'pb2tdelr', 'pbOF', 'pbOF0', 'pbOF1X2P', 'pbOF1X2P0', 'pbOF1X2R', 'pbOF1X2R0'
#                  , 'pbampres', 'pbdelayres', 'pc2tdelr', 'pcOF', 'pcOF0', 'pcOF1X2P', 'pcOF1X2P0', 'pcOF1X2R',
#                  'pcOF1X2R0', 'pcampres', 'pcdelayres', 'pd2tdelr', 'pdOF', 'pdOF0', 'pdOF1X2P', 'pdOF1X2P0',
#                  'pdOF1X2R', 'pdOF1X2R0', 'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
#                  'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF', 'pprimechanOFWK'
#                  , 'pprimechanWK', 'pprimechaniOF', 'prdelWK', 'prpartOF', 'prpartOF1X2', 'prxypartOF', 'prxypartOF1X2',
#                  'psumOF', 'psumOF1X2', 'psumiOF', 'psumiOF1X2', 'psumoOF', 'ptNF', 'ptNF0', 'ptNF0uc', 'ptNFuc', 'ptOF'
#                  , 'ptOF0', 'ptOF0uc', 'ptOF1X2P', 'ptOF1X2P0', 'ptOF1X2P0uc', 'ptOF1X2Puc', 'ptOF1X2R', 'ptOF1X2R0',
#                  'ptOFuc', 'ptftWK_9520', 'pxdelWK', 'pxpartOF', 'pxpartOF1X2', 'pydelWK', 'pypartOF', 'pypartOF1X2',
#                  'qiOF', 'qiOF0', 'qoOF', 'qoOF0', 'qrpartOF', 'qsumOF']
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
rrq_var_names = []
# rrq_var_names = ['paOF', 'paampres', 'padelayres', 'pbOF',
#                  'pbampres', 'pbdelayres', 'pcOF', 'pcampres', 'pcdelayres', 'pdOF',
#                  'pdampres', 'pddelayres', 'pminrtOFWK_10100', 'pminrtOFWK_1040',
#                  'pminrtOFWK_1070', 'pminrtWK_10100', 'pminrtWK_1040', 'pminrtWK_1070', 'pprimechanOF',
#                  'pprimechaniOF', 'prdelWK', 'ptNF',
#                  'ptOF', 'pxdelWK', 'pxpartOF', 'pydelWK', 'pypartOF', 'qiOF', 'qoOF', 'qrpartOF']
new_var_names = ["PXTFPchisq", "PYTFPchisq"]
new_var_inputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
new_var_funcs = [lambda args: (cos(radians(30)) * args[0] + cos(radians(150)) * args[1] + cos(radians(270)) * args[2]),
                 lambda args: (sin(radians(30)) * args[0] + sin(radians(150)) * args[1] + sin(radians(270)) * args[2])]
new_var_info = {"names": [], "inputs": [], "funcs": []}
num_scatter_save_path = os.path.join("../results/files/pca_numscatters.txt")


def do_pca(threshold=0.96, proportion=0.):
    assert 0 <= proportion <= 1
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)
    print(np.shape(train_data))
    print("Doing PCA")
    if proportion != 0:
        selected_train_data = []
        print(np.shape(train_data))
        n = np.shape(train_data)[0]
        indices = np.random.choice(list(range(n)), size=(int(n * proportion)))
        inputs = np.array(train_data)
        data = inputs[indices]
        selected_train_data.extend(data)
    else:
        selected_train_data = train_data

    selected_train_data = np.array(selected_train_data)
    model = PCA()
    model.fit_transform(selected_train_data)
    variance_cumsum = model.explained_variance_ratio_.cumsum()

    n_comp = 0
    variances = []
    for cum_variance in variance_cumsum:
        variances.append(cum_variance)
        if cum_variance >= threshold:
            n_comp += 1
            break
        else:
            n_comp += 1

    print("Done with PCA, got:", n_comp)
    print("Variances: ", variances)

    return n_comp, variances


def get_covariance():
    train_data, train_targets, test_data, test_targets, test_dict, variables = data_loader(rq_var_names, rrq_var_names,
                                                                                           new_var_info,
                                                                                           num_scatter_save_path, det)
    cov = EmpiricalCovariance().fit(train_data)
    for i in range(np.shape(cov.covariance_)[0]):
        print(cov.covariance_[i])

    print("\n", cov.location_)


if __name__ == '__main__':
    n_comp, variances = do_pca(threshold=0.99)
    # get_covariance()
