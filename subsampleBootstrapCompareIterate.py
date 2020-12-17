import h5py
import time
import os, sys
import glob
from functools import reduce
import pickle
import argparse
import pdb
import warnings

import numpy as np
import pandas as pd
import random


from scipy import stats,io
import scipy as sp


#import cluster
#import data_reduction
#import plotting
#import dfc
#import utils
import cpm
from cpm import CPM


import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool


iterNum = str(sys.argv[1])

globalOpdir = 'savepath/iter'+iterNum.zfill(2)

if not os.path.isdir(globalOpdir):
    os.makedirs(globalOpdir)




############# TOC #######################

#### Section 1 ####
#### Load data ####
runSec1 = 1
#### Section 2 ####
#### Train Models on HCP ####
runSec2 = 1
#### Section 3 ####
#### Evaluate Models on left out HCP and PNC ####
runSec3 = 1
#### Section 4 ####
#### Plot single model performance ####
runSec4 = 1
#### Section 5 ####
#### Edge histograms ####
runSec5 = 1
#### Section 6 ####
#### Multiple performance evaluation ####
runSec6 = 1
#### Section 7 ####
#### Plots of multiple performance evaluation ####
runSec7 = 1
#### Section 8 ####
#### Circle plots ####
runSec8 = 0
#### Section 9 ####
#### Evaluate thresholded resample models ####
runSec9 = 1
#### Section 10 ####
#### Plot thresholded resample models performance ####
runSec10 = 1
#### Section 9 ####
#### Evaluate thresholded resample models 2 ####
runSec11 = 1
#### Section 10 ####
#### Plot thresholded resample models performance 2 ####
runSec12 = 1


if runSec1 == 1:
    #### Section 1 ####
    #### Load data ####


    ### Load HCP data
    HCPCorrMats=np.load('path_to/HCP_WM_LR_corrmats.npy',allow_pickle=True).item()
    subsToUse=np.load('path/to/substouse.npy')
    HCPpmat=pd.read_csv('path_to/PMATs_inputtoCPM_oneline.csv',index_col=0) 
    meanAbsMotParams=pd.read_csv('path_to/meanAbsMovementWMLRHCP_dcpminput.csv',index_col=0)


    ### Select random subset of size 800 from total sample of ~860ish
    randIndPath = os.path.join(globalOpdir,'Randinds.npy')

    if not os.path.isfile(randIndPath):

        randinds1=np.arange(0,len(subsToUse))
        random.shuffle(randinds1)

        ## Save random indices and pick subject ids
        np.save(randIndPath,randinds1)

    else:
        randinds1 = np.load(randIndPath, allow_pickle = True)


    subsToUse=subsToUse[randinds1]
    subsToUse = subsToUse[:800]

    ### Load PNC Data
    # PNC Data
    PNCmatFile=io.loadmat('path_to/pncdata.mat') 
    PNCData=PNCmatFile['ipmats']
    PNCDataRes=np.reshape(PNCData,[268**2,788])

    PNCpmatFile=io.loadmat('path_to/pncpmats.mat')  
    PNCpmat=np.squeeze(PNCpmatFile['pmats_pnc'])


    ### Define HCP Variables

    HCPDataSample1 = np.stack([HCPCorrMats['sub'+sub] for sub in subsToUse[:400]])
    HCPDataSample2 = np.stack([HCPCorrMats['sub'+sub] for sub in subsToUse[400:]])

                                         
    HCPabsmot = meanAbsMotParams[subsToUse].values.flatten()
    HCPabsmotSample1= HCPabsmot[:400]
    HCPabsmotSample2= HCPabsmot[400:]

    HCPpmat = HCPpmat[subsToUse].values.flatten()
    HCPpmatSample1 = HCPpmat[:400]
    HCPpmatSample2 = HCPpmat[400:]

    # Reshape arrays
    HCPDataSample1=np.reshape(HCPDataSample1,[HCPDataSample1.shape[0],268**2])
    HCPDataSample2=np.reshape(HCPDataSample2,[HCPDataSample2.shape[0],268**2])



if runSec2 == 1:
    #### Section 2 ####
    #### Train Models on HCP ####


    ## Train Only
    print('Train Only')
    trainPath = os.path.join(globalOpdir,'trainOnly.npy')

    if not os.path.isfile(trainPath):
        trainRes = cpm.train_cpm(HCPDataSample1.T,HCPpmatSample1, corrtype = 'partial', confound=HCPabsmotSample1) 
        np.save(trainPath,trainRes)
    else:
        trainRes = np.load(trainPath, allow_pickle = True)

    ## Splithalf CV
    print('Splithalf')
    splitHalfPath = os.path.join(globalOpdir,'splithalfCV.npy')
    if not os.path.isfile(splitHalfPath):
        splitHalfRes = cpm.run_validate(HCPDataSample1.T,HCPpmatSample1,cvtype = 'splithalf', niters = 100, corrtype = 'partial', confound=HCPabsmotSample1) 

        np.save(splitHalfPath,{'res':splitHalfRes})
    else:
        splitHalfRes = np.load(splitHalfPath, allow_pickle = True).item()

    ## Five fold
    print('Five fold')
    fiveFoldPath = os.path.join(globalOpdir,'5kCV.npy')
    if not os.path.isfile(fiveFoldPath):
        fiveFoldRes = cpm.run_validate(HCPDataSample1.T,HCPpmatSample1,cvtype = '5k', niters = 100, corrtype = 'partial', confound=HCPabsmotSample1)
        np.save(fiveFoldPath,{'res':fiveFoldRes})
    else:
        fiveFoldRes = np.load(fiveFoldPath, allow_pickle = True).item()

    ## Ten fold
    print('Ten Fold')
    tenFoldPath = os.path.join(globalOpdir,'10kCV.npy')
    if not os.path.isfile(tenFoldPath):
        tenFoldRes = cpm.run_validate(HCPDataSample1.T,HCPpmatSample1,cvtype = '10k', niters = 100, corrtype = 'partial', confound=HCPabsmotSample1) 
        np.save(tenFoldPath,{'res':tenFoldRes})
    else:
        tenFoldRes = np.load(tenFoldPath, allow_pickle = True).item()

    ## LOO
    print('LOO')
    looPath = os.path.join(globalOpdir,'looCV.npy')
    if not os.path.isfile(looPath):
        looRes = cpm.run_validate(HCPDataSample1.T,HCPpmatSample1,cvtype = 'LOO', niters = 1, corrtype = 'partial', confound=HCPabsmotSample1) 
        np.save(looPath,{'res':looRes})
    else:
        looRes = np.load(looPath, allow_pickle = True).item()

    ## Bootstrap
    print('Bootstrap')
    bootPath = os.path.join(globalOpdir,'bootstrap.npy')
    if not os.path.isfile(bootPath):
        bootRes = cpm.resampleCPM(HCPDataSample1.T,HCPpmatSample1, 400, replacement = True, iters = 100, corrtype = 'partial', confound=HCPabsmotSample1)
        np.save(bootPath,{'res':bootRes})
    else:
        bootRes = np.load(bootPath, allow_pickle = True).item()
        bootRes = bootRes['res']
    bootedges = np.array(bootRes[1]).mean(axis=0) > 0
    bootmodel = np.mean(bootRes[3],axis=0)

    ## Subsample
    print('Subsample 300')
    sub300Path = os.path.join(globalOpdir,'subsample300.npy')
    if not os.path.isfile(sub300Path):
        sub300Res = cpm.resampleCPM(HCPDataSample1.T,HCPpmatSample1, 300, replacement = False, iters = 100, corrtype = 'partial', confound=HCPabsmotSample1)
        np.save(sub300Path,{'res':sub300Res})
    else:
        sub300Res = np.load(sub300Path, allow_pickle = True).item()
        sub300Res = sub300Res['res']
    sub300edges = np.array(sub300Res[1]).mean(axis=0) > 0
    sub300model = np.mean(sub300Res[3],axis=0)

    ## Subsample
    print('Subsample 200')
    sub200Path = os.path.join(globalOpdir,'subsample200.npy')
    if not os.path.isfile(sub200Path):
        sub200Res = cpm.resampleCPM(HCPDataSample1.T,HCPpmatSample1, 200, replacement = False, iters = 100, corrtype = 'partial', confound=HCPabsmotSample1)
        np.save(sub200Path,{'res':sub200Res})
    else:
        sub200Res = np.load(sub200Path, allow_pickle = True).item()
        sub200Res = sub200Res['res']
    sub200edges = np.array(sub200Res[1]).mean(axis=0) > 0
    sub200model = np.mean(sub200Res[3],axis=0)




if runSec3 == 1:
    #### Section 3 ####
    #### Evaluate Models on left out HCP and PNC ####


    ## Random sample of PNC
    resampleInds = np.random.choice(range(0,787),size=400,replace=False)
    subIpmats =PNCDataRes[:,resampleInds].T
    subPmats = PNCpmat[resampleInds]


    ## Resampled Models HCP & PNC evaluation

    # HCP

    bootRHCP = cpm.apply_cpm(HCPDataSample2,HCPpmatSample2,bootedges,bootmodel,False,False,400)
    sub300RHCP = cpm.apply_cpm(HCPDataSample2,HCPpmatSample2,sub300edges,sub300model,False,False,400)
    sub200RHCP = cpm.apply_cpm(HCPDataSample2,HCPpmatSample2,sub200edges,sub200model,False,False,400)

    # PNC

    bootRPNC = cpm.apply_cpm(subIpmats,subPmats,bootedges,bootmodel,False,False,400)
    sub300RPNC = cpm.apply_cpm(subIpmats,subPmats,sub300edges,sub300model,False,False,400)
    sub200RPNC = cpm.apply_cpm(subIpmats,subPmats,sub200edges,sub200model,False,False,400)


    ## CV Models HCP & PNC evaluation

    def applyAllModels(features,models,testX,testY):
        gather=[]

        for i in range(0,len(models)):
            print(i)
            edges = features[i,:]
            mod = models[i,:]
            perf = cpm.apply_cpm(testX,testY,edges,mod,False,False,400)
            gather.append(perf)

        return np.array(gather)


    # SplitHalf
    posEdgeMasktwoK = np.concatenate(splitHalfRes['res'][0])
    posFitstwoK = np.concatenate(splitHalfRes['res'][5])

    gatherHCP2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,HCPDataSample2,HCPpmatSample2)
    gatherPNC2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmats,subPmats)

    # 5k
    posEdgeMaskfiveK = np.concatenate(fiveFoldRes['res'][0])
    posFitsfiveK = np.concatenate(fiveFoldRes['res'][5])

    gatherHCP5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,HCPDataSample2,HCPpmatSample2)
    gatherPNC5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmats,subPmats)

    #10K
    posEdgeMasktenK = np.concatenate(tenFoldRes['res'][0])
    posFitstenK = np.concatenate(tenFoldRes['res'][5])

    gatherHCP10K = applyAllModels(posEdgeMasktenK,posFitstenK,HCPDataSample2,HCPpmatSample2)
    gatherPNC10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmats,subPmats)

    #loo
    posEdgeMaskloo = np.concatenate(looRes['res'][0])
    posFitsloo = np.concatenate(looRes['res'][5])

    gatherHCPLOO = applyAllModels(posEdgeMaskloo,posFitsloo,HCPDataSample2,HCPpmatSample2)
    gatherPNCLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmats,subPmats)

    #train only
    trainMod = trainRes[0]
    trainEdges = trainRes[2]

    trainRHCP = cpm.apply_cpm(HCPDataSample2,HCPpmatSample2,trainEdges,trainMod,False,False,400)
    trainRPNC = cpm.apply_cpm(subIpmats,subPmats,trainEdges,trainMod,False,False,400)

    ## CV Performance evaluated across all folds
    # 2K
    posBehavRes = np.reshape(splitHalfRes['res'][2],[100,400])
    actBehavRes = np.reshape(splitHalfRes['res'][4],[100,400])

    cvPerf2K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,50)])

    # 5K
    posBehavRes = np.reshape(fiveFoldRes['res'][2],[100,400])
    actBehavRes = np.reshape(fiveFoldRes['res'][4],[100,400])

    cvPerf5K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,50)])

    # 10K
    posBehavRes = np.reshape(tenFoldRes['res'][2],[100,400])
    actBehavRes = np.reshape(tenFoldRes['res'][4],[100,400])

    cvPerf10K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,50)])

    # LOO
    posBehavRes = np.reshape(looRes['res'][2],[400])
    actBehavRes = np.reshape(looRes['res'][4],[400])

    cvPerfloo = np.corrcoef(posBehavRes,actBehavRes)[0,1]
if runSec5 == 1:
    #### Section 5 ####
    #### Edge Histograms ####

    ## Edge plots for resample and CV models
    bootedgesAv = np.array(bootRes[1]).mean(axis=0)
    sub300edgesAv = np.array(sub300Res[1]).mean(axis=0)
    sub200edgesAv = np.array(sub200Res[1]).mean(axis=0)

    posEdgeMasktwoKAv = np.concatenate(splitHalfRes['res'][0]).mean(axis=0)
    posEdgeMaskfiveKAv = np.concatenate(fiveFoldRes['res'][0]).mean(axis=0)
    posEdgeMasktenKAv = np.concatenate(tenFoldRes['res'][0]).mean(axis=0)
    posEdgeMasklooAv = np.concatenate(looRes['res'][0]).mean(axis=0)


    #plt.clf()
    def hplot(ipedges,edgelims,title):

        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        sns.set(style="whitegrid", palette="bright", color_codes=True)
        ax.hist(ipedges[ipedges > 0],range=(0,1))
        ax2.hist(ipedges[ipedges > 0],range=(0,1))
        ax.set_title(title, fontsize=14)
        ax.set_ylim(edgelims[1], edgelims[0])  # outliers only
        ax2.set_ylim(edgelims[3], edgelims[2])  # most of the data
        ax.tick_params(bottom=False)  # don't put tick labels at the top

        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        plt.xlabel('Frequency of occurence', fontsize=12)
        plt.ylabel('Number of edges', fontsize=12)

        sns.despine(ax=ax,left=False,bottom=True)
        sns.despine(ax=ax2,left=False, bottom=False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax2.xaxis.grid(False)
        ax2.yaxis.grid(False)


    plt.figure(figsize=[6,6])
    hplot(bootedgesAv[bootedgesAv > 0],[19000,11000,5000,0],'Bootstrap Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'bootstrapHist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(sub300edgesAv[sub300edgesAv > 0],[19000,11000,5000,0],'Subsample 300 Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'sub300Hist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(sub200edgesAv[sub200edgesAv > 0],[19000,11000,5000,0],'Subsample 200 Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'sub200Hist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(posEdgeMasktwoKAv[posEdgeMasktwoKAv > 0],[19000,11000,5000,0],'Split Half Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'2KHist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(posEdgeMaskfiveKAv[posEdgeMaskfiveKAv > 0],[19000,11000,5000,0],'Five Fold Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'5KHist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(posEdgeMasktenKAv[posEdgeMasktenKAv > 0],[19000,11000,5000,0],'Ten Fold Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'10KHist.png'))
    plt.close()

    plt.figure(figsize=[6,6])
    hplot(posEdgeMasklooAv[posEdgeMasklooAv > 0],[19000,11000,5000,0],'Loo Edge Distribution') 
    plt.savefig(os.path.join(globalOpdir,'looHist.png'))
    plt.close()

    plt.close('all')

    # montage 2KHist.png 5KHist.png 10KHist.png looHist.png sub200Hist.png sub300Hist.png bootstrapHist.png -tile 4x2 -geometry 1000x1000 edgeMontage.png

if runSec6 == 1:
    #### Section 6 ####
    #### Evaluation of models on multiple subsamples of left out HCP and PNC ####


    iterResOppath = os.path.join(globalOpdir,'iterRes.npy')

    if not os.path.isfile(iterResOppath):

        resDict = {}

        for perfIter in range(0,100):

            resDict[perfIter] = {}

            resampleIndsPNC = np.random.choice(range(0,787),size=200,replace=False)
            subIpmatsPNC =PNCDataRes[:,resampleIndsPNC].T
            subPmatsPNC = PNCpmat[resampleIndsPNC]


            resampleIndsHCP = np.random.choice(range(0,400),size=200,replace=False)
            subPmatsHCP = HCPpmatSample2[resampleIndsHCP]
            subIpmatsHCP = HCPDataSample2[resampleIndsHCP,:]


            ## Resampled Models

            # HCP

            bootRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,bootedges,bootmodel,False,False,400)
            sub300RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub300edges,sub300model,False,False,400)
            sub200RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub200edges,sub200model,False,False,400)

            # PNC

            bootRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,bootedges,bootmodel,False,False,400)
            sub300RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub300edges,sub300model,False,False,400)
            sub200RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub200edges,sub200model,False,False,400)


            # SplitHalf
            gatherHCP2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsHCP,subPmatsHCP)
            gatherPNC2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsPNC,subPmatsPNC)

            # 5k
            gatherHCP5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsHCP,subPmatsHCP)
            gatherPNC5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsPNC,subPmatsPNC)

            #10K
            gatherHCP10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsHCP,subPmatsHCP)
            gatherPNC10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsPNC,subPmatsPNC)

            #loo
            gatherHCPLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsHCP,subPmatsHCP)
            gatherPNCLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsPNC,subPmatsPNC)

            #train only
            trainRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,trainEdges,trainMod,False,False,400)
            trainRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,trainEdges,trainMod,False,False,400)


            resDict[perfIter]['bootHCP'] = bootRHCP
            resDict[perfIter]['sub300HCP'] = sub300RHCP
            resDict[perfIter]['sub200HCP'] = sub200RHCP
            resDict[perfIter]['HCP2K'] = gatherHCP2K
            resDict[perfIter]['HCP5K'] = gatherHCP5K
            resDict[perfIter]['HCP10K'] = gatherHCP10K
            resDict[perfIter]['HCPLOO'] = gatherHCPLOO
            resDict[perfIter]['HCPTrain'] = trainRHCP

            resDict[perfIter]['bootPNC'] = bootRPNC
            resDict[perfIter]['sub300PNC'] = sub300RPNC
            resDict[perfIter]['sub200PNC'] = sub200RPNC
            resDict[perfIter]['PNC2K'] = gatherPNC2K
            resDict[perfIter]['PNC5K'] = gatherPNC5K
            resDict[perfIter]['PNC10K'] = gatherPNC10K
            resDict[perfIter]['PNCLOO'] = gatherPNCLOO
            resDict[perfIter]['PNCTrain'] = trainRPNC



        np.save(iterResOppath,resDict)

    else:
        resDict = np.load(iterResOppath, allow_pickle = True).item()





    diffPerfs = resDict[0].keys()

    for resType in diffPerfs:
        exec(resType+'Iter = np.stack([resDict[i]["'+resType+'"] for i in range(0,100)]).flatten()')






#from scipy import stats
#stats.ttest_ind(PNC10KIter, sub200PNCIter, equal_var=False)
#stats.ttest_ind(sub200PNCIter,PNC10KIter, equal_var=False)
#stats.ttest_ind(sub200PNCIter**2,PNC10KIter**2, equal_var=False)
#stats.ttest_ind(bootstrapPNCIter**2,PNC10KIter**2, equal_var=False)
#stats.ttest_ind(bootPNCIter**2,PNC10KIter**2, equal_var=False)
#stats.ttest_ind(bootPNCIter**2,PNClooIter**2, equal_var=False)
#stats.ttest_ind(bootPNCIter**2,PNCLooIter**2, equal_var=False)
#stats.ttest_ind(bootPNCIter**2,PNCLOOIter**2, equal_var=False)





if runSec7 == 1:
    #### Section 7 ####
    #### Performance Plots of multi iterations ####

    ## Correlation boxplot ##
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('CV Performance')
    plt.boxplot([cvPerf2K,cvPerf5K,cvPerf10K,cvPerfloo,np.array([np.nan]),np.array([np.nan]),np.array([np.nan])])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','','','',''])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Performance on left out HCP')
    plt.boxplot([HCP2KIter,HCP5KIter,HCP10KIter,HCPLOOIter,sub200HCPIter,sub300HCPIter,bootHCPIter,HCPTrainIter])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Performance on PNC')
    plt.boxplot([PNC2KIter,PNC5KIter,PNC10KIter,PNCLOOIter,sub200PNCIter,sub300PNCIter,bootPNCIter,PNCTrainIter])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfIter.png'))




    ## R squared boxplot
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('CV Performance')
    plt.boxplot([cvPerf2K**2,cvPerf5K**2,cvPerf10K**2,cvPerfloo**2,np.array([np.nan]),np.array([np.nan]),np.array([np.nan])])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','','','',''])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Performance on left out HCP')
    plt.boxplot([HCP2KIter**2,HCP5KIter**2,HCP10KIter**2,HCPLOOIter**2,sub200HCPIter**2,sub300HCPIter**2,bootHCPIter**2,HCPTrainIter**2])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Performance on PNC')
    plt.boxplot([PNC2KIter**2,PNC5KIter**2,PNC10KIter**2,PNCLOOIter**2,sub200PNCIter**2,sub300PNCIter**2,bootPNCIter**2,PNCTrainIter**2])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfIterSq.png'))




    ## Correlation violinplot
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('CV Performance')
    plt.violinplot([cvPerf2K,cvPerf5K,cvPerf10K,cvPerfloo,np.array([np.nan,np.nan]),np.array([np.nan,np.nan]),np.array([np.nan,np.nan])])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','','','',''])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Performance on left out HCP')
    plt.violinplot([HCP2KIter,HCP5KIter,HCP10KIter,HCPLOOIter,sub200HCPIter,sub300HCPIter,bootHCPIter,HCPTrainIter])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Performance on PNC')
    plt.violinplot([PNC2KIter,PNC5KIter,PNC10KIter,PNCLOOIter,sub200PNCIter,sub300PNCIter,bootPNCIter,PNCTrainIter])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfIterViolin.png'))




    ## R squared violinplot
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('CV Performance')
    plt.violinplot([cvPerf2K**2,cvPerf5K**2,cvPerf10K**2,cvPerfloo**2,np.array([np.nan,np.nan]),np.array([np.nan,np.nan]),np.array([np.nan,np.nan])])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','','','',''])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Performance on left out HCP')
    plt.violinplot([HCP2KIter**2,HCP5KIter**2,HCP10KIter**2,HCPLOOIter**2,sub200HCPIter**2,sub300HCPIter**2,bootHCPIter**2,HCPTrainIter**2])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Performance on PNC')
    plt.violinplot([PNC2KIter**2,PNC5KIter**2,PNC10KIter**2,PNCLOOIter**2,sub200PNCIter**2,sub300PNCIter**2,bootPNCIter**2,PNCTrainIter**2])
    plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
    plt.ylim([-0.1,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfIterSqViolin.png'))

    plt.close('all')

if runSec8 == 1:

    #### Section 8 ####
    #### Montage of circle plots ####
    bootedgesAv = np.array(bootRes[1]).mean(axis=0)
    sub300edgesAv = np.array(sub300Res[1]).mean(axis=0)
    sub200edgesAv = np.array(sub200Res[1]).mean(axis=0)
    posEdgeMasktwoKAv = np.concatenate(splitHalfRes['res'][0]).mean(axis=0)
    posEdgeMaskfiveKAv = np.concatenate(fiveFoldRes['res'][0]).mean(axis=0)
    posEdgeMasktenKAv = np.concatenate(tenFoldRes['res'][0]).mean(axis=0)
    posEdgeMasklooAv = np.concatenate(looRes['res'][0]).mean(axis=0)


    def do_thresh_montage(edges,thresh,opname):
        if not os.path.isfile(opname):
            edges[edges < thresh] = 0
            make_montage(edges,opname)


    if not os.path.isfile(os.path.join(globalOpdir,'montageFigs/')):
        os.makedirs(os.path.join(globalOpdir,'montageFigs/'))

    for i in np.arange(0,1,0.1):
        i = round(i,2)

        do_thresh_montage(bootedgesAv,i,os.path.join(globalOpdir,'montageFigs/bootstrapEdges'+str(round(i,2)).replace('.','')+'.png'))


    plt.close('all')





if runSec9 == 1:
    #### Section 9 ####
    #### Thresholded ensemble model performance ####

    bootedgesAv = np.array(bootRes[1]).mean(axis=0)
    sub300edgesAv = np.array(sub300Res[1]).mean(axis=0)
    sub200edgesAv = np.array(sub200Res[1]).mean(axis=0)

    posEdgeMasktwoKAv = np.concatenate(splitHalfRes['res'][0]).mean(axis=0)
    posEdgeMaskfiveKAv = np.concatenate(fiveFoldRes['res'][0]).mean(axis=0)
    posEdgeMasktenKAv = np.concatenate(tenFoldRes['res'][0]).mean(axis=0)
    posEdgeMasklooAv = np.concatenate(looRes['res'][0]).mean(axis=0)


    threshResOppath = os.path.join(globalOpdir,'threshRes.npy')

    if not os.path.isfile(threshResOppath):

        resDict = {}


        resDict['bootHCP'] = {}
        resDict['sub300HCP'] = {}
        resDict['sub200HCP'] = {}
        #resDict['HCP2K'] = {}
        #resDict['HCP5K'] = {}
        #resDict['HCP10K'] = {}
        #resDict['HCPLOO'] = {}
        #resDict['HCPTrain'] = {}


        resDict['bootPNC'] = {}
        resDict['sub300PNC'] = {}
        resDict['sub200PNC'] = {}






        threshs = list(map(lambda x : round (x,2), np.arange(0,1,0.1)))


        for rk in resDict.keys():
            for thresh in threshs:
                resDict[rk][thresh] = []


        for perfIter in range(0,100):
            resampleIndsPNC = np.random.choice(range(0,787),size=200,replace=False)
            subIpmatsPNC =PNCDataRes[:,resampleIndsPNC].T
            subPmatsPNC = PNCpmat[resampleIndsPNC]


            resampleIndsHCP = np.random.choice(range(0,400),size=200,replace=False)
            subPmatsHCP = HCPpmatSample2[resampleIndsHCP]
            subIpmatsHCP = HCPDataSample2[resampleIndsHCP,:]



            for thresh in threshs:
                ## Resampled Models

                if thresh == 0:
                    bootedgesThresh = bootedgesAv > thresh
                    sub300edgesThresh = sub300edgesAv > thresh
                    sub200edgesThresh = sub200edgesAv > thresh
                else:
                    bootedgesThresh = bootedgesAv >= thresh
                    sub300edgesThresh = sub300edgesAv >= thresh
                    sub200edgesThresh = sub200edgesAv >= thresh
                
                # HCP

                bootRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,bootedgesThresh,bootmodel,False,False,200)
                sub300RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub300edgesThresh,sub300model,False,False,200)
                sub200RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub200edgesThresh,sub200model,False,False,200)

                # PNC

                bootRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,bootedgesThresh,bootmodel,False,False,200)
                sub300RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub300edgesThresh,sub300model,False,False,200)
                sub200RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub200edgesThresh,sub200model,False,False,200)


                # SplitHalf
                #gatherHCP2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsPNC,subPmatsPNC)

                # 5k
                #gatherHCP5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsPNC,subPmatsPNC)

                #10K
                #gatherHCP10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsPNC,subPmatsPNC)

                #loo
                #gatherHCPLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsHCP,subPmatsHCP)
                #gatherPNCLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsPNC,subPmatsPNC)

                #train only
                #trainRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,trainEdges,trainMod,False,False,400)
                #trainRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,trainEdges,trainMod,False,False,400)


                resDict['bootHCP'][thresh].append(bootRHCP)
                resDict['sub300HCP'][thresh].append(sub300RHCP)
                resDict['sub200HCP'][thresh].append(sub200RHCP)

                resDict['bootPNC'][thresh].append(bootRPNC)
                resDict['sub300PNC'][thresh].append(sub300RPNC)
                resDict['sub200PNC'][thresh].append(sub200RPNC)


        np.save(threshResOppath,resDict)

    else:
        resDict = np.load(threshResOppath, allow_pickle = True).item()





    bootHCP = np.stack([resDict['bootHCP'][k] for k in resDict['bootHCP'].keys()])
    sub300HCP = np.stack([resDict['sub300HCP'][k] for k in resDict['sub300HCP'].keys()])
    sub200HCP = np.stack([resDict['sub200HCP'][k] for k in resDict['sub200HCP'].keys()])

    bootPNC = np.stack([resDict['bootPNC'][k] for k in resDict['bootPNC'].keys()])
    sub300PNC = np.stack([resDict['sub300PNC'][k] for k in resDict['sub300PNC'].keys()])
    sub200PNC = np.stack([resDict['sub200PNC'][k] for k in resDict['sub200PNC'].keys()])

if runSec10 == 1:

    #### Section 10 ####
    #### Plotting thresholded ensemble model performance ####
    threshs = list(map(lambda x : round (x,2), np.arange(0,1,0.1)))
    xlabels= ['>='+str(t) for t in threshs]



    ## Correlation boxplot HCP based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap HCP Performance')
    plt.boxplot(bootHCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 HCP Performance')
    plt.boxplot(sub300HCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 HCP Performance')
    plt.boxplot(sub200HCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshHCP.png'))

    ## Correlation boxplot PNC based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap PNC Performance')
    plt.boxplot(bootPNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 PNC Performance')
    plt.boxplot(sub300PNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 PNC Performance')
    plt.boxplot(sub200PNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshPNC.png'))




    ## Correlation boxplot HCP based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap HCP Performance')
    plt.boxplot(bootHCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 HCP Performance')
    plt.boxplot(sub300HCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 HCP Performance')
    plt.boxplot(sub200HCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshHCPRsq.png'))

    ## Correlation boxplot PNC based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap PNC Performance')
    plt.boxplot(bootPNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 PNC Performance')
    plt.boxplot(sub300PNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 PNC Performance')
    plt.boxplot(sub200PNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshPNCRsq.png'))


    plt.close('all')



if runSec11 == 1:
    #### Section 11 ####
    #### Thresholded ensemble model performance 2 ####

    bootedgesAv = np.array(bootRes[1]).mean(axis=0)
    sub300edgesAv = np.array(sub300Res[1]).mean(axis=0)
    sub200edgesAv = np.array(sub200Res[1]).mean(axis=0)

    posEdgeMasktwoKAv = np.concatenate(splitHalfRes['res'][0]).mean(axis=0)
    posEdgeMaskfiveKAv = np.concatenate(fiveFoldRes['res'][0]).mean(axis=0)
    posEdgeMasktenKAv = np.concatenate(tenFoldRes['res'][0]).mean(axis=0)
    posEdgeMasklooAv = np.concatenate(looRes['res'][0]).mean(axis=0)

    threshRes2Oppath = os.path.join(globalOpdir,'threshRes.npy')

    if not os.path.isfile(threshResOppath):
        resDict = {}


        resDict['bootHCP'] = {}
        resDict['sub300HCP'] = {}
        resDict['sub200HCP'] = {}
        #resDict['HCP2K'] = {}
        #resDict['HCP5K'] = {}
        #resDict['HCP10K'] = {}
        #resDict['HCPLOO'] = {}
        #resDict['HCPTrain'] = {}


        resDict['bootPNC'] = {}
        resDict['sub300PNC'] = {}
        resDict['sub200PNC'] = {}





        threshs = list(map(lambda x : round (x,2), np.arange(0,1,0.1)))


        for rk in resDict.keys():
            for thresh in threshs:
                resDict[rk][thresh] = []


        for perfIter in range(0,100):
            resampleIndsPNC = np.random.choice(range(0,787),size=200,replace=False)
            subIpmatsPNC =PNCDataRes[:,resampleIndsPNC].T
            subPmatsPNC = PNCpmat[resampleIndsPNC]


            resampleIndsHCP = np.random.choice(range(0,400),size=200,replace=False)
            subPmatsHCP = HCPpmatSample2[resampleIndsHCP]
            subIpmatsHCP = HCPDataSample2[resampleIndsHCP,:]



            for thresh in threshs:
                ## Resampled Models

                if thresh == 0:
                    bootedgesThresh = (bootedgesAv > thresh) & (bootedgesAv < thresh+0.1)
                    sub300edgesThresh = (sub300edgesAv > thresh) & (sub300edgesAv < thresh+0.1)
                    sub200edgesThresh = (sub200edgesAv > thresh) & (sub200edgesAv < thresh+0.1)
                else:
                    bootedgesThresh = (bootedgesAv >= thresh) & (bootedgesAv < thresh+0.1)
                    sub300edgesThresh = (sub300edgesAv >= thresh) & (sub300edgesAv < thresh+0.1)
                    sub200edgesThresh = (sub200edgesAv >= thresh) & (sub200edgesAv < thresh+0.1)
                
                # HCP

                bootRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,bootedgesThresh,bootmodel,False,False,200)
                sub300RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub300edgesThresh,sub300model,False,False,200)
                sub200RHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,sub200edgesThresh,sub200model,False,False,200)

                # PNC

                bootRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,bootedgesThresh,bootmodel,False,False,200)
                sub300RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub300edgesThresh,sub300model,False,False,200)
                sub200RPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,sub200edgesThresh,sub200model,False,False,200)


                # SplitHalf
                #gatherHCP2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC2K = applyAllModels(posEdgeMasktwoK,posFitstwoK,subIpmatsPNC,subPmatsPNC)

                # 5k
                #gatherHCP5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC5K = applyAllModels(posEdgeMaskfiveK,posFitsfiveK,subIpmatsPNC,subPmatsPNC)

                #10K
                #gatherHCP10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsHCP,subPmatsHCP)
                #gatherPNC10K = applyAllModels(posEdgeMasktenK,posFitstenK,subIpmatsPNC,subPmatsPNC)

                #loo
                #gatherHCPLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsHCP,subPmatsHCP)
                #gatherPNCLOO = applyAllModels(posEdgeMaskloo,posFitsloo,subIpmatsPNC,subPmatsPNC)

                #train only
                #trainRHCP = cpm.apply_cpm(subIpmatsHCP,subPmatsHCP,trainEdges,trainMod,False,False,400)
                #trainRPNC = cpm.apply_cpm(subIpmatsPNC,subPmatsPNC,trainEdges,trainMod,False,False,400)


                resDict['bootHCP'][thresh].append(bootRHCP)
                resDict['sub300HCP'][thresh].append(sub300RHCP)
                resDict['sub200HCP'][thresh].append(sub200RHCP)

                resDict['bootPNC'][thresh].append(bootRPNC)
                resDict['sub300PNC'][thresh].append(sub300RPNC)
                resDict['sub200PNC'][thresh].append(sub200RPNC)


        np.save(threshRes2Oppath,resDict)

    else:
        resDict = np.load(threshRes2Oppath, allow_pickle = True).item()




    bootHCP = np.stack([resDict['bootHCP'][k] for k in resDict['bootHCP'].keys()])
    sub300HCP = np.stack([resDict['sub300HCP'][k] for k in resDict['sub300HCP'].keys()])
    sub200HCP = np.stack([resDict['sub200HCP'][k] for k in resDict['sub200HCP'].keys()])

    bootPNC = np.stack([resDict['bootPNC'][k] for k in resDict['bootPNC'].keys()])
    sub300PNC = np.stack([resDict['sub300PNC'][k] for k in resDict['sub300PNC'].keys()])
    sub200PNC = np.stack([resDict['sub200PNC'][k] for k in resDict['sub200PNC'].keys()])

if runSec12 == 1:

    #### Section 12 ####
    #### Plotting thresholded ensemble model performance 2 ####
    threshs = list(map(lambda x : round (x,2), np.arange(0,1,0.1)))
    xlabels= ['>= '+str(t)+' <'+str(round(t+0.1,2)) for t in threshs]



    ## Correlation boxplot HCP based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap HCP Performance')
    plt.boxplot(bootHCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 HCP Performance')
    plt.boxplot(sub300HCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 HCP Performance')
    plt.boxplot(sub200HCP.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshHCP2.png'))

    ## Correlation boxplot PNC based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap PNC Performance')
    plt.boxplot(bootPNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 PNC Performance')
    plt.boxplot(sub300PNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 PNC Performance')
    plt.boxplot(sub200PNC.T)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.6])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshPNC2.png'))




    ## Correlation boxplot HCP based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap HCP Performance')
    plt.boxplot(bootHCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 HCP Performance')
    plt.boxplot(sub300HCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 HCP Performance')
    plt.boxplot(sub200HCP.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshHCPRsq2.png'))

    ## Correlation boxplot PNC based performance
    plt.figure(figsize=[12,10])
    plt.subplot(3,1,1)
    plt.title('Bootstrap PNC Performance')
    plt.boxplot(bootPNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,2)
    plt.title('Subsample 300 PNC Performance')
    plt.boxplot(sub300PNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.subplot(3,1,3)
    plt.title('Subsample 200 PNC Performance')
    plt.boxplot(sub200PNC.T**2)
    plt.xticks(range(1,11),xlabels)
    plt.ylim([0,0.3])
    plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'perfSubsampleThreshPNCRsq2.png'))


    plt.close('all')







