from __future__ import division
import numpy
from numpy import asarray
import scipy
import scipy.io as sio
import os.path
from os import listdir
import scipy, scipy.stats
import pandas as pd
import pickle 
import h5py
from importlib import reload
import glob
import IPython.core.debugger as ipdb
import pdb

import logging
# set DEBUG for everything
#logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('matplotlib')
# set WARNING for Matplotlib
logger.setLevel(logging.WARNING)
#-------
#import tabulate
#-------
import LoadData
import Raw_data as RD

def NR_ScF(Energy, material='Ge'):
    #Lindhard Model
    if material=='Ge':
        massNumber = 72.64 
        atomicNumber = 32
        
    epsilon = 0.0115 * Energy * atomicNumber**(-7/3)
    # where energy is in eV
    k = 0.133 * atomicNumber**(2/3) * massNumber**(-1/2)                                                                                                                   
    g = 3*epsilon**0.15 + 0.7*epsilon**0.6 + epsilon                                                                                
    return (1+k*g)/(k*g)

def TrueEns(Energy, inter=0, Vbias=4):
    #Energy: in eV
    #inter:  interaction type  (0: nuclear recoil   1: electron recoil)
    if inter == 0:
        NRScale = NR_ScF(Energy)
    elif inter == 1:
        NRScale = 1
    trueQEn  = Energy/NRScale
    truePhEn = Energy*(1-1/NRScale)    
    elecHoleCreation = 2.96 #eV Pehl, NIM 59(1), pp. 45?55, 1968
    truePhEn = truePhEn + (1+elecHoleCreation/Vbias)*trueQEn
    return trueQEn, truePhEn

def getTrue(inputfile,evlist,zipNum):
    print("\nGetting truth information from particle hits file...")
    try:
        thF=sio.loadmat(inputfile)
        filetypename='Older Mat format'
        tempInfo=pd.DataFrame( { 'evnum': thF['EV'][:,0], 
                                'evnum_store': thF['EV'][:,0], 
                                'Type {}'.format(zipNum): thF['Type'][:,0],
                                'PTSIMRecoilEnergy {}'.format(zipNum): thF['D3'][:,0]/1e3,
                                'PTSIMAvgX {}'.format(zipNum): thF['X3'][:,0],  
                                'PTSIMAvgY {}'.format(zipNum): thF['Y3'][:,0],
                                'PTSIMAvgZ {}'.format(zipNum): thF['Z3'][:,0]  }).set_index('evnum')
    except NotImplementedError:
        thF=h5py.File(inputfile,'r')
        filetypename='v7.3 Mat format'
        tempInfo=pd.DataFrame( { 'evnum': thF['EV'][0,:], 'evnum_store': thF['EV'][0,:], 
                                'Type {}'.format(zipNum): thF['Type'][0,:],
                                'PTSIMRecoilEnergy {}'.format(zipNum): thF['D3'][0,:]/1e3,
                                'PTSIMAvgX {}'.format(zipNum): thF['X3'][0,:],  
                                'PTSIMAvgY {}'.format(zipNum): thF['Y3'][0,:],
                                'PTSIMAvgZ {}'.format(zipNum): thF['Z3'][0,:]  }).set_index('evnum')
    print("Found {}".format(filetypename))
    trueInfo = pd.DataFrame( {'evnum': numpy.insert(numpy.sort(numpy.unique(tempInfo.index.get_level_values(0))),0,0) } )
    
    trueInfo['PTSIMRecoilEnergy {}'.format(zipNum)] = 0
    trueInfo['PTSIMAvgX {}'.format(zipNum)] = 0
    trueInfo['PTSIMAvgY {}'.format(zipNum)] = 0
    trueInfo['PTSIMAvgZ {}'.format(zipNum)] = 0
    trueInfo['NScatters {}'.format(zipNum)] = 0
    trueInfo['ExpQEn {}'.format(zipNum)] = 0
    trueInfo['ExpPhEn {}'.format(zipNum)] = 0
    
    kk=0
    for thev in evlist:
        thevEV = trueInfo.loc[int(thev),'evnum']
        for thv in ['PTSIMRecoilEnergy','PTSIMAvgX','PTSIMAvgY','PTSIMAvgZ','NScatters']: 
            if 'Avg' in thv:
                if type(tempInfo.loc[thevEV,thv+' {}'.format(zipNum)]) != numpy.float64:
                    tempVals = tempInfo.loc[thevEV,thv+' {}'.format(zipNum)].values*weights
                    tempVal = tempVals.sum()
                else: 
                    tempVal = tempInfo.loc[thevEV,thv+' {}'.format(zipNum)]
                #print('Average:', tempVal)
            elif 'NScatters' == thv:
                if type(tempInfo.loc[thevEV,'PTSIMRecoilEnergy {}'.format(zipNum)]) != numpy.float64:
                    tempVal = len(tempInfo.loc[thevEV,'PTSIMRecoilEnergy {}'.format(zipNum)].values)
                else:
                    tempVal = 1
            else:
                tempVal = tempInfo.loc[thevEV,thv+' {}'.format(zipNum)].sum()
                if 'Recoil' in thv:
                    if type(tempInfo.loc[thevEV,thv+' {}'.format(zipNum)]) != numpy.float64:
                        weights = tempInfo.loc[thevEV,thv+' {}'.format(zipNum)].values/tempVal
                    else: weights = 1
                    if type(tempInfo.loc[thevEV,['PTSIMRecoilEnergy {}'.format(zipNum),'Type {}'.format(zipNum)]].values[0]) == numpy.float64:
                        thscat = tempInfo.loc[thevEV,['PTSIMRecoilEnergy {}'.format(zipNum),'Type {}'.format(zipNum)]].values
                        if int(thscat[1]) in [11,22]:
                            #If the scattering particle was a photon, electron, or positron (so Electron Recoil)
                            intertype = 1
                        elif thscat[1] >= 1000.0:
                            #If the scattering particle was something heavy (so a nuclear recoil)
                            intertype = 0
                        thQE, thPhE = TrueEns( thscat[0]*1e3, inter=intertype )
                        trueInfo.loc[int(thev),'ExpQEn {}'.format(zipNum)] = trueInfo.loc[int(thev),'ExpQEn {}'.format(zipNum)] + thQE
                        trueInfo.loc[int(thev),'ExpPhEn {}'.format(zipNum)] = trueInfo.loc[int(thev),'ExpPhEn {}'.format(zipNum)] + thPhE
                    elif type(tempInfo.loc[thevEV,['PTSIMRecoilEnergy {}'.format(zipNum),'Type {}'.format(zipNum)]].values[0]) == numpy.ndarray:
                        for thscat in tempInfo.loc[thevEV,['PTSIMRecoilEnergy {}'.format(zipNum),'Type {}'.format(zipNum)]].values:
                            if int(thscat[1]) in [11,22]:
                                intertype = 1
                            elif thscat[1] >= 1000.0:
                                intertype = 0
                            thQE, thPhE = TrueEns( thscat[0]*1e3, inter=intertype )
                            trueInfo.loc[int(thev),'ExpQEn {}'.format(zipNum)] = trueInfo.loc[int(thev),'ExpQEn {}'.format(zipNum)] + thQE
                            trueInfo.loc[int(thev),'ExpPhEn {}'.format(zipNum)] = trueInfo.loc[int(thev),'ExpPhEn {}'.format(zipNum)] + thPhE  
                            #used to have thev-1 above; not sure about that
                    else:
                        print('WARNING! Type problem!')
            trueInfo.loc[int(thev),thv+' {}'.format(zipNum)] = tempVal
        if kk%20 == 0:
            print("...{}/{}".format(kk,len(evlist)), end="\r")
        kk+=1
    print("Done...{}/{}".format(kk,len(evlist)), end="\r")
    return trueInfo.loc[evlist]

def get_full_data(do_SS, do_qCol, do_raw, do_bats, 
                  sim_prefix, detectors,
                  DMCdata, DMCsubdirs, 
                  qcolpaths, col_evs,
                  inputfiles, trueFiles,
                  raw_files,rawnames,series_nums,raw_DF_saveFiles, raw_readDFs, raw_saveDFs,
                  eventrqs, calibevtqs, uncalibrqs, dmcuncalibrqs, calibrrqs,
                  n_samples=2048, chan_list=(0,), concatcol = False, calcInput = False, saveInput = False, 
                  binproc = [0], concatsamps=False,
                  debug=False):
    #The arguments are lists of labels, paths for CDMSBats, Charge Collection, Hits output, three calibration files, 
    #   list of indices that have been binary-writer processed, detector to use 
    
    #The main sections below are:
    # 1. Setup
    # 2. Reading CDMSBats files
    # 3. Reading DMC charge collection
    # 4. Merging CDMSBats and Charge Collection data
    # 5. Get Particle Hits data
    # 6. Get Raw data
    # 7. Combine samples, do calculations
    
    #***************************** SECTION 1: SETUP ***************************

        #Get calibration values
    #with open(calib_files[0],'rb') as f:
    #    calvals = pickle.load(f)
    #with open(calib_files[1],'rb') as f:
    #    postcalvals = pickle.load(f)
    #with open(calib_files[2],'rb') as f:
    #    bariumcalib = pickle.load(f)
        
    #Some constants
    ptNFcalib=42820000      #(keV/amp)
    ADCampNorm=1638400000   #(adcbins/amp)
    delt=1.6e-6 
    nbin=4096
    delnu= 1/(delt*nbin)
    nu= delnu*numpy.linspace(0,nbin-1,nbin)   
    
    reindex_const=50000

    #evcuts=['cGoodDCOffset_v53']
    #evbools=[True]

    zdet=detectors#4
    rununiques=True
    runcommon=False

    if debug:
        print("DEBUG: Setup done. Enter 'c' to continue.")
        pdb.set_trace()
    
    #***************************** SECTION 2: READ CDMSBATS FILES ***************************
    #the list allSIMs is the main output of this entire function; each list entry is the data for a sample

    # In Load Data function:
    #           load_data(base_location, data_location, cut_location, filename_template, detectors,
    #                     eventrqs,uncalibrqs,calibrrqs,calibevtqs,first=None,last=None)
    if do_bats:
        thop=0

        #reload(LoadData)

    #    getRedTable=True
        print ("Loading rrq data...",)

        allSIMs=[]
        DataDetCuts=[]
        DataEvCuts=[]
        for ic in range(0,len(DMCdata)):
            print ("Loading data...{}/{}...".format(ic+1,len(DMCdata)),end="\r")
            #dop is a switch for binary-processed stuff; it changes which inputs are passed to load_data
            if ic in binproc: dop=1
            else: dop=0
            thDMC=pd.DataFrame()
            thDMC, thevcuts, thdetcuts = LoadData.load_data(DMCdata[ic],
                             DMCsubdirs[dop], #subdirectories inside DMCdata
                             'nonsense',#'cuts/current/ba_permitted_Sept2013', 
                             ['merge*.root','merge*.root'][dop], #file template
                             detectors,
                             eventrqs,[dmcuncalibrqs,uncalibrqs][dop],calibrrqs,calibevtqs,
                                                  None, None, 
                                                  None, None, 
                                                  [None,None][dop],[None,None][dop], 
                                                  apply_evcuts=False, debug=False )

            allSIMs.append(pd.DataFrame())
            allSIMs[ic] = pd.concat([thDMC]) 
            allSIMs[ic].index = range(len(allSIMs[ic]))
            if (ic==0 and thop ==0) or ic<10:
                    DataEvCuts.append(pd.DataFrame(dtype=bool))
                    DataEvCuts[ic]= pd.concat([thevcuts])
                    DataEvCuts[ic].index = range(len(DataEvCuts[ic]))
                    DataDetCuts.append(pd.DataFrame(dtype=bool))
                    DataDetCuts[ic]= pd.concat([thdetcuts])
                    DataDetCuts[ic].index = range(len(DataDetCuts[ic]))
            else:
                DataEvCuts.append(pd.DataFrame(dtype=bool))
                DataEvCuts[ic]= DataEvCuts[0]
                DataDetCuts.append(pd.DataFrame(dtype=bool))
                DataDetCuts[ic]= DataDetCuts[0]
        print( "Done loading rrq data          ")
        
        if debug:
            print("DEBUG: CDMSBats section done. Enter 'c' to continue.")
            pdb.set_trace()

    #************************ SECTION 3: READ CHARGE COLLECTION DATA FROM THE DMC ***********************

    # The list qCol is created here; it's eventually merged onto allSims

    if do_qCol:
    
        droplist=['pu','pd','effpu','effpd','Ecolpu','Ecolpd']

        qCol=[]
        ic =0
        for qcolpath in qcolpaths:
            #True = combine multiple files together in the qcolbase directory (and save that)
            #False = just read in the previously-saved qCol file
            if concatcol:
                print('Combining charge collection files')
                evstep=col_evs[0]
                numevs=col_evs[1]
                njobs=numevs/evstep
                print ('Loading ev {} to {}...'.format(0,numevs), end='\r')
                for thi in range(0,int(njobs)):
                    fev=thi*evstep+1
                    lev=(thi+1)*evstep
                    print ('...Loading ev {} to {}...'.format(fev,lev), end='\r')
                    if thi==0:
                        thqCol=pd.read_hdf(qcolpath+'qCol_%d-%d.hdf'%(fev,lev),'qCol')
                        thqCol.drop(droplist, axis=1)
                        #thePos=pd.read_hdf(qcolpath+'ePos_%d-%d.hdf'%(fev,lev),'ePos')
                        #thhPos=pd.read_hdf(qcolpath+'hPos_%d-%d.hdf'%(fev,lev),'hPos')
                    else:
                        ththqCol=pd.read_hdf(qcolpath+'qCol_%d-%d.hdf'%(fev,lev),'qCol').drop(droplist,axis=1)
                        thqCol=pd.concat([thqCol,ththqCol],sort=False)
                        del ththqCol
                        #thePos=pd.concat([thePos,pd.read_hdf(qcolpath+'ePos_%d-%d.hdf'%(fev,lev),'ePos')],sort=False)
                        #thhPos=pd.concat([thhPos,pd.read_hdf(qcolpath+'hPos_%d-%d.hdf'%(fev,lev),'hPos')],sort=False)
                print ("\nDone")
                thqCol.to_hdf(qcolpath+'qCol_full.hdf','qCol')
                #thePos.to_hdf(qcolpath+'ePos_full.hdf','ePos')
                #thhPos.to_hdf(qcolpath+'hPos_full.hdf','hPos')

            else:
                print('Reading saved charge collection file')
                thqCol=pd.read_hdf(qcolpath+'qCol_full.hdf','qCol')
                #thePos=pd.read_hdf(qcolpath+'ePos_full.hdf','ePos')
                #thhPos=pd.read_hdf(qcolpath+'hPos_full.hdf','hPos')
            qCol.append(thqCol)

            print("Length of this sample's qCol: ",len(thqCol))
            del thqCol
            ic +=1
            
        if debug:
            print("DEBUG: Charge collection section done. Enter 'c' to continue.")
            pdb.set_trace()
        
    #************************ SECTION 4: MERGE CDMSBATS AND CHARGE COLLECTION DATA ***********************

    # Fix event numbering to merge samples later
    for ii in range(1,len(allSIMs)): 
        #Each sample gets its event numbers multiplied proportionally to its index in allSIMs
        reix=reindex_const*ii
        allSIMs[ii]['EventNumber'] = allSIMs[ii]['EventNumber'].values + reix
        if do_qCol:
            qCol[ii]['EventNumber'] = qCol[ii].index.get_level_values(0).astype(int) + reix 
            qCol[ii] = qCol[ii].set_index('EventNumber', append=False)
    if do_qCol:
        for ii in range(len(allSIMs)):
            print( "Sorting...", end='\r')
            sortix=qCol[ii].index.values.argsort()
            qCol[ii]=qCol[ii].iloc[sortix]
            print( "Sorted....qCol sample ",ii)
            #sortix=ePos.index.values.argsort()
            #ePos=ePos.iloc[sortix]
            #print "ePos...",
            #sortix=hPos.index.values.argsort()
            #hPos=hPos.iloc[sortix]
            #print "hPos...Done"
    print("")
    
    for ii in range(len(allSIMs)): 
        print("Number of events in sample {}:".format(ii))
        print(len(allSIMs[ii]))
    print("")

    #Check event numbers
    for ii in range(len(allSIMs)):
        if do_qCol:
            print ("qCol event numbers:")
            print (qCol[ii].index.values.astype(int) )
        print("CDMSBats event numbers:")
        if ii in binproc: print (allSIMs[ii]['EventNumber'].values.astype(int))
        else: print (allSIMs[ii]['PTSIMEventNumber {}'.format(zdet[ii])].values.astype(int))

    #Dump randoms/noise
    ucut={}
    for ii in range(len(allSIMs)):
        print ("Sim: ",sim_prefix[ii])
        if ii in binproc: 
            #Event category 0 means triggered events--i.e. not randoms
            tmpcut= allSIMs[ii]['EventCategory'].values == 0 
            allSIMs[ii]=allSIMs[ii][tmpcut]
            print ("\t\tDumping Noise Events (binaryprocessed)...",len(tmpcut),len(allSIMs[ii]))
        else: 
            u, indices = numpy.unique(allSIMs[ii]['PTSIMEventNumber {}'.format(zdet[ic])].values, return_index=True)
            #u, indices = numpy.unique(allSIMs[ic]['EventNumber'].values, return_index=True)
            indices=numpy.asarray(indices)
            ucut[ii]=indices
            if rununiques:
                allSIMs[ii]=allSIMs[ii].iloc[ucut[ii]]
            print ("\t\tDumping Noise Events (rootprocessed)...",len(allSIMs[ii]))
    if rununiques: rununiques=False


    #find events mutual to rrqs and charge collection data and cut down to those
    if do_qCol:
        for ic in range(0,len(allSIMs)):
            print ("Sim: ", sim_prefix[ic])
            print( "Getting events contained in DMC sample and Charge Collection Info........")
            if ic in binproc:
                goodevs=numpy.asarray( numpy.in1d( qCol[ic].index.values.astype(int), 
                                                  allSIMs[ic]['EventNumber'].values.astype(int)  ) )
            else: 
                goodevs=numpy.asarray( numpy.in1d( qCol[ic].index.values.astype(int), 
                                                  allSIMs[ic]['PTSIMEventNumber {}'.format(zdet[ic])].values.astype(int)  ) )
            print ("\tqCol-->SIM: number of entries:", len(goodevs)," passing common events:",sum(goodevs))
            qCol[ic]=qCol[ic][goodevs]
            print ("\t\tfinal size of samples: ",len(qCol[ic]),len(allSIMs[ic]))
            if ic in binproc:
                goodevs=numpy.asarray( numpy.in1d( allSIMs[ic]['EventNumber'].values.astype(int), 
                                                  qCol[ic].index.values.astype(int) ) )
            else:
                goodevs=numpy.asarray( numpy.in1d( allSIMs[ic]['PTSIMEventNumber {}'.format(zdet[ic])].values.astype(int), 
                                                  qCol[ic].index.values.astype(int) ) )
            print ("\tSIM-->qCol: number of entries:", len(goodevs)," passing common events:",sum(goodevs))
            allSIMs[ic]=allSIMs[ic][goodevs]
            print ("\t\tfinal size of samples: ",len(qCol[ic]),len(allSIMs[ic]) )

        #make sure events are in correct order between allSIMs and qCol?
        for ii in range(len(allSIMs)):
            if ii in binproc:
                qCol[ii] = qCol[ii].loc[allSIMs[ii]['EventNumber'].values]

        for ic in range(len(allSIMs)):
            print("Sim: ", sim_prefix[ic])
            for jj in range(len(qCol[ic].index.values)):
                if ic in binproc:
                    if qCol[ic].index.values[jj] != allSIMs[ic]['EventNumber'].values[jj]:
                        print( "Events Not in order, index:",jj,qCol[ic].index.values[jj] ,
                              allSIMs[ic]['EventNumber'].values[jj])
                else:
                    if qCol[ic].index.values[jj] != allSIMs[ic]['PTSIMEventNumber {}'.format(zdet[ic])].values[jj]:
                        print( "Events Not in order, index:",jj,qCol[ic].index.values[jj],
                              allSIMs[ic]['PTSIMEventNumber {}'.format(zdet[ic])].values[jj])        

    else: print("Charge collection skipped! Just checking CDMSBats event numbers...")                    
                        
    #order/index by event numbers 
    for ic in range(0,len(allSIMs)):
        allSIMs[ic]['oldindex'] = allSIMs[ic].index.values
        if ic in binproc:
            allSIMs[ic]=allSIMs[ic].set_index('EventNumber')
        else: 
            allSIMs[ic]=allSIMs[ic].set_index('PTSIMEventNumber {}'.format(zdet[ic]))
        if do_qCol:
            #****combine qCol into allSIMs****
            allSIMs[ic]=pd.concat([allSIMs[ic],qCol[ic]],axis=1)   

    if do_qCol:
        #append det num into names
        for ic in range(0,len(allSIMs)):
            allSIMs[ic]=allSIMs[ic].rename(columns={ ths: ths+' %d'%zdet[ic] for ths in qCol[ic].columns.values })

    #scale energies to keV
    for ic in range(0,len(allSIMs)):
        print ("Sim: ",sim_prefix[ic])
        for thv in ['Ecolqiu {}'.format(zdet[ic]), 'Ecolqou {}'.format(zdet[ic]), 
                    'Ecolqid {}'.format(zdet[ic]), 'Ecolqod {}'.format(zdet[ic]), 
                    'Esim {}'.format(zdet[ic])]: # 'Ecolpu {}'.format(zdet[ic]),'Ecolpd {}'.format(zdet[ic]),
            if thv in allSIMs[ic].columns:
                allSIMs[ic][thv]=1e-3*allSIMs[ic][thv]
            
    if debug:
        print("DEBUG: CDMSBats/QCol combination done. Enter 'c' to continue.")
        pdb.set_trace()

    #************************ SECTION 5: GET PARTICLE HITS DATA ***********************
    #Calculate or read in Hits information
    
    print("Getting SourceSim/hits information...")

    if do_SS:
        for ic in range(0,len(allSIMs)):
            if calcInput:
                inputfile=inputfiles[ic]
                if ic in binproc:
                    reindex=[0,reindex_const][ic]   #same reindex number from earlier
                    thDF=getTrue(inputfile,allSIMs[ic].index.get_level_values(0).values.astype(int)-reindex,zdet[ic])
                    thDF['evnum']=thDF.index.get_level_values(0)+reindex
                    thDF=thDF.set_index('evnum')
            else:
                print("Reading saved truth information file")
                thDF = pd.read_hdf(trueFiles[ic],'input')
            print( "Events in hits data and rq/rrq data: ", len(thDF), len(allSIMs[ic]))        
            #Combine the Hits information into allSIMs
            allSIMs[ic]=pd.concat([allSIMs[ic],thDF],axis=1)

            if saveInput: 
                thDF.to_hdf(trueFiles[ic],'input')  
                
        if debug:
            print("DEBUG: SourceSim section done. Enter 'c' to continue.")
            pdb.set_trace()
            
    #************************ SECTION 6: GET RAW FILE DATA ***********************    
    if do_raw:
        
        print("Getting raw file data...")
        
        #where the raw files are stored
        datapaths=raw_files

        rnames=rawnames
        yearmonth=''
        seriess=series_nums

        #where to save the results
        storedDfs=raw_DF_saveFiles
        readDfs=raw_readDFs
        saveDFs=raw_saveDFs

        #chanlist seems hard-coded after this! Any '4's can probably change to len(chanlist)
        #...Oh, wait, this is probably specifying just FETs, right? Others may have different N_samples
        chanlist = chan_list
        N_samples=n_samples

        Evs=[]

        kk=0    
        for datapath in datapaths:
            rname=rnames[kk]
            readDf=readDfs#[kk]
            saveDF=saveDFs#[kk]
            series=seriess[kk]

            if readDf:
                print("Reading saved Raw data DataFrame from: {}".format(storedDfs[kk]))
                tEvs = pd.read_hdf(storedDfs[kk],'Evs')
            else:
                print("Creating new Raw data DataFrame...")
                tEvs=pd.DataFrame()
                for thser in series:

                    filelist=sorted(glob.glob(datapath+rname+'_F*.gz'))
                    print( len(filelist),filelist[0])

                    print ("Reading file...{}/{}".format(0,len(filelist)),end="\r")
                    SerEvs=pd.DataFrame()
                    nn=0
                    skipseries=False
                    for thfile in filelist:
                        print ("Reading file...{}/{}".format(nn+1,len(filelist)),end="\r")
                        try: df = RD.read_file(thfile,detlist=detectors,chanlist=chanlist,n_samples=N_samples)
                        except: 
                            print( "Problems reading dump ",nn," in series: ",thser)
                            print ("\t",thfile)
                            skipseries=False
                            nn+=1
                            continue
                        #df = RD.read_file(thfile,detlist=detectors,chanlist=chanlist,n_samples=N_samples)
                        seriesnumber=''.join(thser.split('_'))
                        serdf=pd.DataFrame(numpy.ones(len(df.index))*int(seriesnumber), columns=['series number'])
                        df=pd.concat([df,serdf],axis=1)
                        #df = df.set_index(['series number','event number','channel number'])#,'detector number'])
                        SerEvs=pd.concat([SerEvs,df],axis=0)
                        del df, serdf
                        nn+=1
                    if not skipseries:
                        tEvs=pd.concat([tEvs,SerEvs],axis=0)
                    del SerEvs
                    print ("Done Reading dumps in series",thser," {}/{}".format(nn,len(filelist)))

                if saveDF:
                    print("Saving raw data to: {}".format(storedDfs[kk]))
                    tEvs.to_hdf(storedDfs[kk],'Evs')

            if kk ==0: #Used to always concatenate everything. Better to sim-index?
                #Evs=tEvs
                Evs.append(tEvs)
            else: 
                tEvs['event number'] = tEvs['event number'].values + reindex_const*kk
                tEvs=tEvs[1000*len(chanlist):] 
                #Evs=pd.concat([Evs,tEvs],axis=0)
                Evs.append(tEvs)

            del tEvs

            kk+=1

        for ic in range(0,len(Evs)):
            #Make sure we have types we can work with and reindex
            Evs[ic]['series number']=Evs[ic]['series number'].values.astype(int)
            Evs[ic]['event number'] = Evs[ic]['event number'].values.astype(int)
            Evs[ic]['channel number'] = Evs[ic]['channel number'].values.astype(int)
            Evs[ic] = Evs[ic].set_index(['series number','event number','detector number','channel number'])

            #Remove data from the first file, which is the noise dump? Assumes 500 entries with 4 channels?
            # Also remove event category and type
            Evs[ic]=Evs[ic][500*len(chanlist):].drop(['event category','event type'],axis=1)

            #get the indices for raw events that match our allSIMs entries
            thix = []
            allIX = allSIMs[ic].index.values
            for kk in range(len(allIX)):
                for thc in chanlist:
                    #print(Evs)
                    #index with series number, event number (matching allSIMs),det number, and channel number
                    thix.append((Evs[ic].index.values[0][0],allIX[kk],zdet[ic],thc))
            #and get those events
            Evs[ic]=Evs[ic].loc[thix]

            #Removes detNum and chanNum indices; leaves series and eventNum indices
            EvsIX = Evs[ic][::len(chanlist)].index.droplevel(-1).droplevel(-1)

            #Do our two datasets match up? (Have the same length, at least...?)
            print("Events in allSIMs: {}\nEvents in EvsIX: {}".format(len(allSIMs[ic]),len(EvsIX)))

            traces = Evs[ic].values.reshape(-1,1,len(chanlist),2048)

            for kk in range(len(EvsIX)):
                try:
                    #Evs[ic]=Evs[ic].sort_index()
                    traces[kk,0]=Evs[ic].loc[EvsIX[kk]].loc[zdet[ic]].values
                except:
                    print("Something went wrong with the trace indices! Sample ", kk)
                    break
            #traces = traces[ptp(traces[:,0], axis=-1) < 95]
            #traces = traces.reshape(-1, 1, 2, 2, 2048)
            # Dimensions are (event, zip, S1 vs S2, I vs O, time)
            #print(numpy.shape(traces))

            #Center at 0?
            #traces = traces - traces[:,:,:,:,0:500].mean(axis=-1,keepdims=True)
            traces = traces - traces[:,:,:,0:500].mean(axis=-1,keepdims=True)

            #not sure about this...
            print("Finding max values of traces...")
            #maxvals = traces[:,0,:,0,1000:1100].max(axis=-1)-traces[:,0,:,0,1000:1100].min(axis=-1)
            maxvals = traces[:,0,:,1000:1100].max(axis=-1)-traces[:,0,:,1000:1100].min(axis=-1)
            allSIMs[ic]['QI1Max {}'.format(zdet[ic])] = 0
            allSIMs[ic]['QI2Max {}'.format(zdet[ic])] = 0

            kk=0
            for thix in EvsIX: 
                allSIMs[ic].loc[thix[1], 'QI1Max {}'.format(zdet[ic])] = maxvals[kk,0]
                allSIMs[ic].loc[thix[1], 'QI2Max {}'.format(zdet[ic])] = maxvals[kk,1]    
                kk+=1
        if debug:
            print("DEBUG: Raw data section done. Enter 'c' to continue.")
            pdb.set_trace()
    
    #************************ SECTION 7: COMBINE SAMPLES AND DO CALCULATIONS ***********************
    
    #CONCATenate SAMPleS into one, if multiple were passed
    if concatsamps:
        print("Combining samples...")
        # concat into index 0 
        allSIMs = [ pd.concat([allSIMs[0],allSIMs[1]], axis=0) ]
        qCol  = [pd.concat([qCol[0],qCol[1]],axis=0) ]
        if zdet[0] != zdet[1]:
            print("WARNING!: combining samples for different detectors!")

    print("Calculating rrqs...")
    #Calculate rrqs
    for ic in  range(0,len(allSIMs)):
        if ('qi1OFr {}'.format(zdet[ic]) not in allSIMs[ic].columns) and ('qi1OF {}'.format(zdet[ic]) in allSIMs[ic].columns):
            allSIMs[ic]['qi1OFr {}'.format(zdet[ic])] = allSIMs[ic]['qi1OF {}'.format(zdet[ic])].values
            allSIMs[ic]['qi2OFr {}'.format(zdet[ic])] = allSIMs[ic]['qi2OF {}'.format(zdet[ic])].values
            allSIMs[ic]['qo1OFr {}'.format(zdet[ic])] = allSIMs[ic]['qo1OF {}'.format(zdet[ic])].values
            allSIMs[ic]['qo2OFr {}'.format(zdet[ic])] = allSIMs[ic]['qo2OF {}'.format(zdet[ic])].values
            allSIMs[ic]['qimeanr {}'.format(zdet[ic])]=(allSIMs[ic]['qi1OFr {}'.format(zdet[ic])].values + allSIMs[ic]['qi2OFr {}'.format(zdet[ic])].values)/2
        if (('QIS1OFvoltsr {}'.format(zdet[ic]) not in allSIMs[ic].columns) 
            and ('QIS1OFvolts {}'.format(zdet[ic]) in allSIMs[ic].columns)):
            allSIMs[ic]['QIS1OFvoltsr {}'.format(zdet[ic])] = allSIMs[ic]['QIS1OFvolts {}'.format(zdet[ic])].values
            allSIMs[ic]['QOS1OFvoltsr {}'.format(zdet[ic])] = allSIMs[ic]['QOS1OFvolts {}'.format(zdet[ic])].values
            allSIMs[ic]['QIS2OFvoltsr {}'.format(zdet[ic])] = allSIMs[ic]['QIS2OFvolts {}'.format(zdet[ic])].values
            allSIMs[ic]['QOS2OFvoltsr {}'.format(zdet[ic])] = allSIMs[ic]['QOS2OFvolts {}'.format(zdet[ic])].values
            
            allSIMs[ic]['QIS1OFvolts {}'.format(zdet[ic])] = allSIMs[ic]['QIS1OFvoltsr {}'.format(zdet[ic])].values*1e6
            allSIMs[ic]['QOS1OFvolts {}'.format(zdet[ic])] = allSIMs[ic]['QOS1OFvoltsr {}'.format(zdet[ic])].values*1e6
            allSIMs[ic]['QIS2OFvolts {}'.format(zdet[ic])] = allSIMs[ic]['QIS2OFvoltsr {}'.format(zdet[ic])].values*1e6
            allSIMs[ic]['QOS2OFvolts {}'.format(zdet[ic])] = allSIMs[ic]['QOS2OFvoltsr {}'.format(zdet[ic])].values*1e6

        #This section involves calibration files that were made on the spot...
        #for thii in [1,2]:
            #allSIMs[0]['qi%dOFc {}'.format(zdet[ic])%thii] = allSIMs[0]['qi%dOFr {}'.format(zdet[ic])%thii].values*\
            #                                    (1/(params[2]*(params[0]*allSIMs[0]['PTSIMAvgZ {}'.format(zdet[ic])].values+params[1])+1))
            #allSIMs[0]['qi%dOF {}'.format(zdet[ic])%thii] = allSIMs[0]['qi%dOFc {}'.format(zdet[ic])%thii].values*\
            #                                    (1/postcalvals['Qi%dDelE'%thii])
            #allSIMs[0]['qo%dOF {}'.format(zdet[ic])%thii] = allSIMs[0]['qo%dOFr {}'.format(zdet[ic])%thii].values*(1/(params[2]+1))
            #allSIMs[0]['qi%dOFb {}'.format(zdet[ic])%thii] = allSIMs[0]['qi%dOFr {}'.format(zdet[ic])%thii].values*\
            #                                    (1/bariumcalib['Qi%drDelE'%thii])
            #allSIMs[0]['qo%dOFb {}'.format(zdet[ic])%thii] = allSIMs[0]['qo%dOFr {}'.format(zdet[ic])%thii].values*(1/(params[2]+1))
        #allSIMs[ic]['qimean {}'.format(zdet[ic])]=(allSIMs[ic]['qi1OF {}'.format(zdet[ic])].values + allSIMs[ic]['qi2OF {}'.format(zdet[ic])].values)/2
        #allSIMs[ic]['qimeanb {}'.format(zdet[ic])]=(allSIMs[ic]['qi1OFb {}'.format(zdet[ic])].values + allSIMs[ic]['qi2OFb {}'.format(zdet[ic])].values)/2
        #allSIMs[ic]['qimeanc {}'.format(zdet[ic])]=(allSIMs[ic]['qi1OFc {}'.format(zdet[ic])].values + allSIMs[ic]['qi2OFc {}'.format(zdet[ic])].values)/2    

        #allSIMs[ic]['qi1OF {}'.format(zdet[ic])] = 1/calvals['qi1OF']*(allSIMs[ic]['qi1OFr {}'.format(zdet[ic])].values) # - calvals['qi1OFoffset']) #-caloffset['Qi1DelE']/1e3
        #allSIMs[ic]['qi2OF {}'.format(zdet[ic])] = 1/calvals['qi2OF']*(allSIMs[ic]['qi2OFr {}'.format(zdet[ic])].values) # - calvals['qi2OFoffset']) #-caloffset['Qi2DelE']/1e3
        #allSIMs[ic]['qo1OF {}'.format(zdet[ic])] = 1/calvals['qi1OF']*(allSIMs[ic]['qo1OFr {}'.format(zdet[ic])].values) # - calvals['qi1OFoffset']) #-caloffset['Qi1DelE']/1e3
        #allSIMs[ic]['qo2OF {}'.format(zdet[ic])] = 1/calvals['qi2OF']*(allSIMs[ic]['qo2OFr {}'.format(zdet[ic])].values) # - calvals['qi2OFoffset']) #-caloffset['Qi2DelE']/1e3

        if 'plukeqOF {}'.format(zdet[ic]) in allSIMs[ic].columns:
            allSIMs[ic]['RecoilE {}'.format(zdet[ic])]= allSIMs[ic]['ptNF {}'.format(zdet[ic])].values-allSIMs[ic]['plukeqOF {}'.format(zdet[ic])].values  
            allSIMs[ic]['ytNF {}'.format(zdet[ic])]=allSIMs[ic]['qsummaxOF {}'.format(zdet[ic])].values/allSIMs[ic]['RecoilE {}'.format(zdet[ic])].values
        if 'qsum1OF {}'.format(zdet[ic]) in allSIMs[ic].columns and 'qsum2OF {}'.format(zdet[ic]) in allSIMs[ic].columns and 'QS1OFchisq {}'.format(zdet[ic]) in allSIMs[ic].columns and 'QS2OFchisq {}'.format(zdet[ic]) in allSIMs[ic].columns:
            allSIMs[ic]['QSUMchisq {}'.format(zdet[ic])]=numpy.where( 
                allSIMs[ic]['qsum1OF {}'.format(zdet[ic])].values >= allSIMs[ic]['qsum2OF {}'.format(zdet[ic])].values ,
                allSIMs[ic]['QS1OFchisq {}'.format(zdet[ic])].values,
                allSIMs[ic]['QS2OFchisq {}'.format(zdet[ic])].values  )
        if 'qsum1OF {}'.format(zdet[ic]) in allSIMs[ic].columns and 'QS1OFdelay {}'.format(zdet[ic]) in allSIMs[ic].columns and 'QS2OFdelay {}'.format(zdet[ic]) in allSIMs[ic].columns:
            allSIMs[ic]['QSUMdelay {}'.format(zdet[ic])]=numpy.where( 
                allSIMs[ic]['qsum1OF {}'.format(zdet[ic])].values >= allSIMs[ic]['qsum2OF {}'.format(zdet[ic])].values , 
                allSIMs[ic]['QS1OFdelay {}'.format(zdet[ic])].values ,
                allSIMs[ic]['QS2OFdelay {}'.format(zdet[ic])].values  )
        if 'QS1OFchisq {}'.format(zdet[ic]) in allSIMs[ic].columns and 'QS2OFchisq {}'.format(zdet[ic]) in allSIMs[ic].columns:
            allSIMs[ic]['QIchisq {}'.format(zdet[ic])]=(allSIMs[ic]['QS1OFchisq {}'.format(zdet[ic])].values
                                                        +allSIMs[ic]['QS2OFchisq {}'.format(zdet[ic])].values)/2
        if 'PTOFdelay {}'.format(zdet[ic]) in allSIMs[ic].columns and max(allSIMs[ic]['PTOFdelay {}'.format(zdet[ic])].values)  < 1.0 :  
            allSIMs[ic]['PTOFdelay {}'.format(zdet[ic])] = allSIMs[ic]['PTOFdelay {}'.format(zdet[ic])].values/1.6e-6
        if 'QS1OFdelay {}'.format(zdet[ic]) in allSIMs[ic].columns and max(allSIMs[ic]['QS1OFdelay {}'.format(zdet[ic])].values) < 1.0 : 
            allSIMs[ic]['QS1OFdelay {}'.format(zdet[ic])] = allSIMs[ic]['QS1OFdelay {}'.format(zdet[ic])].values/0.8e-6
        if 'QS2OFdelay {}'.format(zdet[ic]) in allSIMs[ic].columns and max(allSIMs[ic]['QS2OFdelay {}'.format(zdet[ic])].values) < 1.0 : 
            allSIMs[ic]['QS2OFdelay {}'.format(zdet[ic])] = allSIMs[ic]['QS2OFdelay {}'.format(zdet[ic])].values/0.8e-6
        if ic > -1 and False:
            if do_SS:
                allSIMs[ic]['ExpQEn {}'.format(zdet[ic])] = 0
                allSIMs[ic]['ExpPhEn {}'.format(zdet[ic])] = 0
                allSIMs[ic]['ExpQEn {}'.format(zdet[ic])], allSIMs[ic]['ExpPhEn {}'.format(zdet[ic])] = tuple(thE/1e3 for thE in TrueEns(allSIMs[ic]['PTSIMRecoilEnergy {}'.format(zdet[ic])].values*1e3, inter=1) ) 
                allSIMs[ic]['PhDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpPhEn {}'.format(zdet[ic])].values 
                                                            - allSIMs[ic]['ptNF {}'.format(zdet[ic])].values)
                allSIMs[ic]['QDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                           - allSIMs[ic]['qsummaxOF {}'.format(zdet[ic])].values)
                #allSIMs[ic]['QiDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                            - allSIMs[ic]['qimean {}'.format(zdet[ic])].values)
                #allSIMs[ic]['QicDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                             - allSIMs[ic]['qimeanc {}'.format(zdet[ic])].values)
                #allSIMs[ic]['QibDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                             - allSIMs[ic]['qimeanb {}'.format(zdet[ic])].values)
                allSIMs[ic]['QirDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                             - allSIMs[ic]['qimeanr {}'.format(zdet[ic])].values)
                allSIMs[ic]['Qi1DelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                             - allSIMs[ic]['qi1OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['Qi2DelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                             - allSIMs[ic]['qi2OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['Qi1rDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                              - allSIMs[ic]['qi1OFr {}'.format(zdet[ic])].values)
                allSIMs[ic]['Qi2rDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                                                              - allSIMs[ic]['qi2OFr {}'.format(zdet[ic])].values)
                #allSIMs[ic]['Qi1bDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                              - allSIMs[ic]['qi1OFb {}'.format(zdet[ic])].values)
                #allSIMs[ic]['Qi2bDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                              - allSIMs[ic]['qi2OFb {}'.format(zdet[ic])].values)
                #allSIMs[ic]['Qi1cDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                              - allSIMs[ic]['qi1OFc {}'.format(zdet[ic])].values)
                #allSIMs[ic]['Qi2cDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values 
                #                                              - allSIMs[ic]['qi2OFc {}'.format(zdet[ic])].values)
                allSIMs[ic]['RecDelE {}'.format(zdet[ic])]=-(allSIMs[ic]['PTSIMRecoilEnergy {}'.format(zdet[ic])].values
                                                             -allSIMs[ic]['RecoilE {}'.format(zdet[ic])].values)
                allSIMs[ic]['PhDelEE {}'.format(zdet[ic])]=allSIMs[ic]['PhDelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpPhEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QDelEE {}'.format(zdet[ic])]=allSIMs[ic]['QDelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QiDelEE {}'.format(zdet[ic])]=allSIMs[ic]['QiDelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QibDelEE {}'.format(zdet[ic])]=allSIMs[ic]['QibDelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['Qi1DelEE {}'.format(zdet[ic])]=allSIMs[ic]['Qi1DelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['Qi2DelEE {}'.format(zdet[ic])]=allSIMs[ic]['Qi2DelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['RecDelEE {}'.format(zdet[ic])]=allSIMs[ic]['RecDelE {}'.format(zdet[ic])].values/allSIMs[ic]['PTSIMRecoilEnergy {}'.format(zdet[ic])].values
                allSIMs[ic]['YLind {}'.format(zdet[ic])]=1/NR_ScF(allSIMs[ic]['PTSIMRecoilEnergy {}'.format(zdet[ic])].values*1e3)
                allSIMs[ic]['PTSIMAvgR %d'%zdet[ic]]=numpy.sqrt(allSIMs[ic]['PTSIMAvgX %d'%zdet[ic]].values**2 + \
                                                        allSIMs[ic]['PTSIMAvgY %d'%zdet[ic]].values**2)
                allSIMs[ic]['PTSIMAvgPhi %d'%zdet[ic]]= numpy.arctan2(allSIMs[ic]['PTSIMAvgY %d'%zdet[ic]].values,
                                                                  allSIMs[ic]['PTSIMAvgX %d'%zdet[ic]].values)
                allSIMs[ic]['PTSIMAvgR2 %d'%zdet[ic]] = numpy.square(allSIMs[ic]['PTSIMAvgR %d'%zdet[ic]].values)
                if do_qCol:
                    allSIMs[ic]['SDelE {}'.format(zdet[ic])] = allSIMs[ic]['Esim {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                
            if do_qCol:
                allSIMs[ic]['EcolS1 {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values + allSIMs[ic]['Ecolqou {}'.format(zdet[ic])].values
                allSIMs[ic]['EcolS2 {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values + allSIMs[ic]['Ecolqod {}'.format(zdet[ic])].values
                allSIMs[ic]['effs1 {}'.format(zdet[ic])] =  allSIMs[ic]['effqiu {}'.format(zdet[ic])].values + allSIMs[ic]['effqou {}'.format(zdet[ic])].values
                allSIMs[ic]['effs2 {}'.format(zdet[ic])] =  allSIMs[ic]['effqid {}'.format(zdet[ic])].values + allSIMs[ic]['effqod {}'.format(zdet[ic])].values
                allSIMs[ic]['normeffiu {}'.format(zdet[ic])] =  allSIMs[ic]['effqiu {}'.format(zdet[ic])].values/0.9844
                allSIMs[ic]['normeffou {}'.format(zdet[ic])] =  allSIMs[ic]['effqou {}'.format(zdet[ic])].values/0.9844
                allSIMs[ic]['normeffid {}'.format(zdet[ic])] =  allSIMs[ic]['effqid {}'.format(zdet[ic])].values/0.9841
                allSIMs[ic]['normeffod {}'.format(zdet[ic])] =  allSIMs[ic]['effqod {}'.format(zdet[ic])].values/0.9841

            if ('Ecolqiur {}'.format(zdet[ic]) not in allSIMs[ic].columns) and do_qCol and do_SS:
                allSIMs[ic]['QcolDelES1 {}'.format(zdet[ic])] = allSIMs[ic]['EcolS1 {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['Ecolqiur {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values
                allSIMs[ic]['Ecolqidr {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDeliur {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiur {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelidr {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqidr {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['Ecolqour {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqou {}'.format(zdet[ic])].values
                allSIMs[ic]['Ecolqodr {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqod {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelour {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqour {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelodr {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqodr {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values

                #for thii in ['u','d']:
                    #params=calvals['QcolDeli%cr'%thii]
                    #allSIMs[0]['Ecolqi%cc {}'.format(zdet[ic])%thii] = allSIMs[0]['Ecolqi%cr {}'.format(zdet[ic])%thii].values*\
                    #                                    (1/(params[2]*(params[0]*allSIMs[0]['PTSIMAvgZ {}'.format(zdet[ic])].values+params[1])+1))
                    #allSIMs[0]['Ecolqi%c {}'.format(zdet[ic])%thii] = allSIMs[0]['Ecolqi%cc {}'.format(zdet[ic])%thii].values*\
                    #                                    (1/postcalvals['QcolDeli%c'%thii])
                    #allSIMs[0]['Ecolqi%cb {}'.format(zdet[ic])%thii] = allSIMs[0]['Ecolqi%cr {}'.format(zdet[ic])%thii].values*\
                    #                                (1/bariumcalib['QcolDeli%cr'%thii])
                    #allSIMs[0]['Ecolqo%cb {}'.format(zdet[ic])%thii] = allSIMs[0]['Ecolqo%cr {}'.format(zdet[ic])%thii].values*\
                    #                                (1/bariumcalib['QcolDeli%cr'%thii])
                #allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])] = 1/0.9844*allSIMs[ic]['Ecolqiur {}'.format(zdet[ic])].values
                #allSIMs[ic]['Ecolqid {}'.format(zdet[ic])] = 1/0.9841*allSIMs[ic]['Ecolqidr {}'.format(zdet[ic])].values
                #allSIMs[ic]['Ecolimean {}'.format(zdet[ic])] = (allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values + allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values)/2
                allSIMs[ic]['Ecolimeanr {}'.format(zdet[ic])] = (allSIMs[ic]['Ecolqiur {}'.format(zdet[ic])].values 
                                                                 + allSIMs[ic]['Ecolqidr {}'.format(zdet[ic])].values)/2
                #allSIMs[ic]['Ecolimeanc {}'.format(zdet[ic])] = (allSIMs[ic]['Ecolqiuc {}'.format(zdet[ic])].values 
                #                                                 + allSIMs[ic]['Ecolqidc {}'.format(zdet[ic])].values)/2
                #allSIMs[ic]['Ecolimeanb {}'.format(zdet[ic])] = (allSIMs[ic]['Ecolqiub {}'.format(zdet[ic])].values 
                #                                                 + allSIMs[ic]['Ecolqidb {}'.format(zdet[ic])].values)/2


                #allSIMs[ic]['QcolDeliub {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiub {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDeliuc {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiuc {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDeliu {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelidb {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqidb {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelidc {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqidc {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelid {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelES1 {}'.format(zdet[ic])] = allSIMs[ic]['EcolS1 {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelES2 {}'.format(zdet[ic])] = allSIMs[ic]['EcolS2 {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelE {}'.format(zdet[ic])] = allSIMs[ic]['Ecolimean {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelEb {}'.format(zdet[ic])] = allSIMs[ic]['Ecolimeanb {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelEc {}'.format(zdet[ic])] = allSIMs[ic]['Ecolimeanc {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolDelEr {}'.format(zdet[ic])] = allSIMs[ic]['Ecolimeanr {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolDelE {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDeliuEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolDeliu {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolDelidEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolDelid {}'.format(zdet[ic])].values/allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values

                #allSIMs[ic]['QcolSDeliu {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values - allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolSDelid {}'.format(zdet[ic])] = allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values - allSIMs[ic]['Esim {}'.format(zdet[ic])].values

                allSIMs[ic]['QcolSDelES1 {}'.format(zdet[ic])] = allSIMs[ic]['EcolS1 {}'.format(zdet[ic])].values - allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                allSIMs[ic]['QcolSDelES2 {}'.format(zdet[ic])] = allSIMs[ic]['EcolS2 {}'.format(zdet[ic])].values - allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolSDelE {}'.format(zdet[ic])] = allSIMs[ic]['Ecolimean {}'.format(zdet[ic])].values - allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolSDelEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolSDelE {}'.format(zdet[ic])].values/allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolSDeliuEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolSDeliu {}'.format(zdet[ic])].values/allSIMs[ic]['Esim {}'.format(zdet[ic])].values
                #allSIMs[ic]['QcolSDelidEE {}'.format(zdet[ic])]=allSIMs[ic]['QcolSDelid {}'.format(zdet[ic])].values/allSIMs[ic]['Esim {}'.format(zdet[ic])].values

                #allSIMs[ic]['QcolEffDeliu {}'.format(zdet[ic])]=allSIMs[ic]['QcolDeliuEE {}'.format(zdet[ic])].values - (allSIMs[ic]['normeffiu {}'.format(zdet[ic])].values-1)
                #allSIMs[ic]['QcolEffDelid {}'.format(zdet[ic])]=allSIMs[ic]['QcolDelidEE {}'.format(zdet[ic])].values - (allSIMs[ic]['normeffid {}'.format(zdet[ic])].values-1)
                #allSIMs[ic]['QcolEffSDeliu {}'.format(zdet[ic])]=allSIMs[ic]['QcolSDeliuEE {}'.format(zdet[ic])].values - (allSIMs[ic]['normeffiu {}'.format(zdet[ic])].values-1)
                #allSIMs[ic]['QcolEffSDelid {}'.format(zdet[ic])]=allSIMs[ic]['QcolSDelidEE {}'.format(zdet[ic])].values - (allSIMs[ic]['normeffid {}'.format(zdet[ic])].values-1)
            if do_SS:
                allSIMs[ic]['QDelES1 {}'.format(zdet[ic])] = allSIMs[ic]['qsum1OF {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
                allSIMs[ic]['QDelES2 {}'.format(zdet[ic])] = allSIMs[ic]['qsum2OF {}'.format(zdet[ic])].values - allSIMs[ic]['ExpQEn {}'.format(zdet[ic])].values
            if do_qCol:
                allSIMs[ic]['OFcolDelS1 {}'.format(zdet[ic])]=-(allSIMs[ic]['EcolS1 {}'.format(zdet[ic])].values -allSIMs[ic]['qsum1OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['OFcolDelS2 {}'.format(zdet[ic])]=-(allSIMs[ic]['EcolS2 {}'.format(zdet[ic])].values -allSIMs[ic]['qsum2OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['OFcolDeliu {}'.format(zdet[ic])]=-(allSIMs[ic]['Ecolqiu {}'.format(zdet[ic])].values -allSIMs[ic]['qi1OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['OFcolDelid {}'.format(zdet[ic])]=-(allSIMs[ic]['Ecolqid {}'.format(zdet[ic])].values -allSIMs[ic]['qi2OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['OFcolDelou {}'.format(zdet[ic])]=-(allSIMs[ic]['Ecolqou {}'.format(zdet[ic])].values -allSIMs[ic]['qo1OF {}'.format(zdet[ic])].values)
                allSIMs[ic]['OFcolDelod {}'.format(zdet[ic])]=-(allSIMs[ic]['Ecolqod {}'.format(zdet[ic])].values -allSIMs[ic]['qo2OF {}'.format(zdet[ic])].values)
                #allSIMs[ic]['OFcolDelimean {}'.format(zdet[ic])]=allSIMs[ic]['qimean {}'.format(zdet[ic])].values - allSIMs[ic]['Ecolimean {}'.format(zdet[ic])].values
            
    print("Done!")
    return allSIMs,traces

