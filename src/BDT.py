import uproot
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def getnumscatters (initpath, write=True):
    init = loadmat(initpath)
    # Get event number for each scatter
    evofscatter  = init["EV"][:,0]+1 # +1 here because starts at 0
    # Get detector for each scatter
    detofscatter = init["DT"][:,0]
    # Get energy for each scatter
    energyofscatter = init["D3"][:,0]
    # Energies of each scatter, to remove very weak scatters
    scatterenergies = {} 
    
    lastindx = 0
    initialev = evofscatter[lastindx]
    
    while lastindx < len(evofscatter)-1:
        # found boolean used so the entire list is not read each time
        found = False
        # Initialize new dict entry when current ev is greater than current maxscatterev
        maxscatterev = 0
        # Start at lastindx so only necessary indices are read
        for i in range(lastindx,len(evofscatter)):
            # if evofscatter[i] >= 98000:
            # print(evofscatter[i])
            if evofscatter[i] == initialev:
                if initialev > maxscatterev:
                    scatterenergies[initialev] = []
                    maxscatterev = initialev
                # If event of scatter matches event number AND detector matches,
                # increment scatter number by one
                if detofscatter[i] == det:
                    scatterenergies[initialev].append(energyofscatter[i])
                found = True
                # If it's the last scatter, set lastindx so the while loop ends
                if i == len(evofscatter)-1:
                    lastindx = i
            elif found:
                lastindx = i
                initialev = evofscatter[i]
                break
            
    numscatters = {} # Output dict
    singlescatters = {} # 1 if single scatter, 0 if multiple
    for ev in scatterenergies:
        numscatters[ev]=0
        maxenergy = max(scatterenergies[ev])
        for energy in scatterenergies[ev]:
            if energy > 0.01*maxenergy:
                numscatters[ev] += 1
        singlescatters[ev] = int(numscatters[ev]==1)
                  
    # Write to file if requested. True by default
    if write:
        scatterout = open("numscatters.txt","w")
        scatterout.write("EventNumber NumberOfScatters\n")
        for ev in numscatters:
            scatterout.write("%d %d\n"%(ev,numscatters[ev]))
        scatterout.close()
        
    return numscatters, singlescatters


def getbranches (merge, branches, tree=None, normalize=True):
    if tree is None:
        tree = merge
    
    evsraw = merge["PTSIMEventNumber"].array().astype(int)
    
    rawbranches = {}
    for branch in branches:
        rawbranches[branch] = tree[branch].array()
        
    output = []
    
    maxvals = {}
    minvals = {}
    
    for branch in branches:
        maxvals[branch] = 0.
        minvals[branch] = 0.
    
    for i in range(len(evsraw)):
        ev = evsraw[i]
        # Some repeats in event number; continue if repeat
        if ev <= len(output):
            continue
        # Add new entry ouput
        output.append({})
        output[-1]["EV"] = ev
        # Add desired variables to output
        for branch in branches:
            output[-1][branch]=rawbranches[branch][i]
            if output[-1][branch] > maxvals[branch]:
                maxvals[branch] = output[-1][branch]
            if output[-1][branch] < minvals[branch]:
                minvals[branch] = output[-1][branch]
    
    if normalize:
        for v in range(len(output)):
            for branch in branches:   
                if output[v][branch] == minvals[branch]:
                    output[v][branch] = -9
                else:
                    output[v][branch] = np.log((output[v][branch]+np.abs(minvals[branch]))/(maxvals[branch]+np.abs(minvals[branch])))
    return output


def addtovariables (variables, name, toadd):
    for i in range(len(variables)):
        variables[i][name] = toadd[variables[i]["EV"]]
    return variables


def mergevariables (var1, var2, names2):
    for i in range(len(var1)):
        for name in names2:
            var1[i][name] = var2[i][name]
    return var1


def calculatevariable (variables, name, invarnames, invars, function):
    for i in range(len(variables)):
        eventdict = invars[i]
        invarlist = []
        for invarname in invarnames:
            invarlist.append(eventdict[invarname])
        outvar = function(invarlist)
        variables[i][name] = outvar
    return variables


def cutenergy (variables, energies, cut=10):
    newvars = []
    newenergies = []
    for v in range(len(variables)):
        entry = variables[v]
        energy = energies[v]
        if energy["ptNF"] < cut:
            newvars.append(entry)
            newenergies.append(energy)
    return newvars, newenergies


def generatefitmatrix (variables, featurenames, targetname, fraction, energies):
    matrix = []
    targets = []
    
    testmatrix = []
    testtargets = []
    
    testdict = []
    
    for v in range(len(variables)):
        entry = variables[v]
        energy = energies[v]["ptNF"]
        sample = []
        sampledict = {}
        sampledict["ptNF"] = energy
        for featurename in featurenames:
            sample.append(entry[featurename])
            sampledict[featurename]=entry[featurename]
        if len(matrix) < fraction*len(variables):
            matrix.append(sample)
            targets.append(entry[targetname])
        else:
            testmatrix.append(sample)
            testdict.append(sampledict)
            testtargets.append(entry[targetname])
    return matrix, targets, testmatrix, testtargets, testdict


def doBDT(rqvarnames, rrqvarnames, newvarinfo, calibpath, mergepath, initpath, savepath, det=14):
    # Loading in data from files
    calib = uproot.open(calibpath)["rrqDir"]["calibzip%d"%(det)]
    # calib = uproot.open(calibpath)

    merge = uproot.open(mergepath)["rqDir"]["zip%d"%(det)]
        
    variables = getbranches(merge, rqvarnames, merge)
    variables = mergevariables(variables,getbranches(merge, rrqvarnames, calib), rrqvarnames)
    
    scatters,singlescatter = getnumscatters(initpath)
    variables = addtovariables(variables, "Single?", singlescatter)
    
    energies = getbranches(merge, ["ptNF"], calib, normalize=False)
    variables,energies = cutenergy(variables, energies, 20.)
    
    if len(newvarinfo["names"]) != 0:
        for n in range(len(newvarinfo["names"])):
            name = newvarinfo["names"][n]
            invarnames = newvarinfo["inputs"][n]
            func = newvarinfo["funcs"][n]
            
            invars = getbranches(merge,invarnames,merge)
            variables = calculatevariable(variables,name,invarnames,invars,func)
    
    # BDT time!
    rng = np.random.RandomState(1)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),algorithm="SAMME",n_estimators=200)
    
    matrix, targets, testmatrix, testtargets, testdict = generatefitmatrix(variables,
                                                                           rqvarnames+rrqvarnames+newvarinfo["names"],
                                                                           "Single?", 0.8, energies)
    bdt.fit(matrix, targets)
    
    print(bdt.feature_importances_)
    
    predictions = bdt.predict(testmatrix)
    scores = bdt.decision_function(testmatrix)
    numtests = len(scores)
    
    splitscores = [[], []]
    for i in range(len(predictions)):
        splitscores[testtargets[i]].append(scores[i])
    
    return scores, splitscores, testmatrix, testtargets, testdict, variables[0].keys()


def plotoutput(scores,splitscores,testmatrix,testtargets,testdict):    
    plt.hist(splitscores, 50, histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single"], range=(-1, 0.3))
    plt.legend(loc=2)
    plt.xlabel("BDT Decision Function")
    plt.savefig(savepath+"scores.png")
    plt.show()
    
    tsingle = []
    tmulti = []
    fsingle = []
    fmulti = []
    purity = []
    
    cuts = np.linspace(-1, 1, num=1000)
    
    bestcut = 0
    for cut in cuts:
        tsingle.append(0)
        tmulti.append(0)
        fsingle.append(0)
        fmulti.append(0)
        purity.append(0)
        
        for i in range(len(splitscores[0])):
            if splitscores[0][i] < cut:
                tmulti[-1] += 1
            else:
                fsingle[-1] += 1
        for j in range(len(splitscores[1])):
            if splitscores[1][j] < cut:
                fmulti[-1] += 1
            else:
                tsingle[-1] += 1
                
        singletot = tsingle[-1] + fmulti[-1]
        multitot = tmulti[-1] + fsingle[-1]
        if singletot == 0:
            tsingle[-1] = 1.
            fsingle[-1] = 1.
        elif multitot == 0:
            tsingle[-1] = 0.
            fsingle[-1] = 0.
        else:
            tsingle[-1] = 1.*tsingle[-1]/singletot
            fsingle[-1] = 1.*fsingle[-1]/multitot
    
    plt.clf()
    plt.plot(fsingle, tsingle)
    line = np.linspace(0, 1, 50)
    rocsum = 0
    for i in range(len(fsingle)-1):
        dx = fsingle[i]-fsingle[i+1]
        rocsum += dx*(tsingle[i]-fsingle[i])
        
    print(rocsum)
        
    plt.plot(line, line, "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(savepath+"roc.png")
    plt.show()
    plt.clf()
    
    oldspectrum = [[], [], []]
    newspectrum = [[], [], []]
    for i in range(len(scores)):
        score = scores[i]
        oldspectrum[testtargets[i]].append(testdict[i]["ptNF"])
        oldspectrum[2].append(testdict[i]["ptNF"])
        if score > -0.1:
            newspectrum[testtargets[i]].append(testdict[i]["ptNF"])
            newspectrum[2].append(testdict[i]["ptNF"])
                    
    plt.hist(oldspectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single", "All"])
    plt.xlim(0, 4)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath+"old_spectrum.png")
    plt.show()
    plt.clf()
    plt.hist(newspectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                               "All"])
    plt.xlim(0, 4)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath+"new_spectrum.png")
    plt.show()
    
    plt.hist(oldspectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                               "All"])
    plt.xlim(0, 4)
    plt.ylim(0, 50)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath+"old_spectrum_zoomed.png")
    plt.show()
    plt.clf()
    plt.hist(newspectrum, 30, range=(0, 4), histtype='step', linewidth=2.0, fill=False, label=["Multiple", "Single",
                                                                                               "All"])
    plt.xlim(0, 4)
    plt.ylim(0, 50)
    plt.xlabel("PTNF (keV)")
    plt.legend()
    plt.savefig(savepath+"new_spectrum_zoomed.png")
    plt.show()
    
# Basic info and paths
det = 14
rqvarnames = ["PTNFchisq"]
rrqvarnames = ["pxpartOF1X2", "pypartOF1X2", "prpartOF1X2", "pxdelWK", "pydelWK", "prdelWK"]
newvarnames = ["PXTFPchisq", "PYTFPchisq"]
newvarinputs = [["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"], ["PDTFPchisq", "PBTFPchisq", "PCTFPchisq"]]
newvarfuncs = [lambda args: (cos(radians(30))*args[0] + cos(radians(150))*args[1] + cos(radians(270))*args[2]),
               lambda args: (sin(radians(30))*args[0] + sin(radians(150))*args[1] + sin(radians(270))*args[2])]
newvarinfo = {"names": [], "inputs": [], "funcs": []}

# calibpath = "calib_LibSimProdv5-6_pn_Sb_T5Z2.root"
calibpath = "../data/calib_LibSimProdv5-4_pn_Sb_T5Z2.root"
mergepath = "../data/merge_LibSimProdv5-4_pn_Sb_T5Z2.root"
initpath = "../data/PhotoNeutronDMC_InitialTest10K_jswfix.mat"
savepath = "figs/bdt_all_scatters/"

if __name__ == '__main__':

    scores, splitscores, testmatrix, testtargets, testdict, varnames = doBDT(rqvarnames, rrqvarnames, newvarinfo,
                                                                             calibpath, mergepath, initpath, savepath)
    print(varnames)
    plotoutput(scores, splitscores, testmatrix, testtargets, testdict)
