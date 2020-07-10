import time
import os, sys
import glob
import argparse
import pdb
import warnings

import numpy as np
import pandas as pd
import random


from scipy import stats,io
import scipy as sp

import cpm



iterNum = str(sys.argv[1])

#### Load data
HCPCorrMats=np.load('PathToHCPWorkingMemoryFCMats',allow_pickle=True).item()
subsToUse=np.load('substouse.npy')
HCPpmat=pd.read_csv('PathToHCPfIQ',index_col=0) 
meanAbsMotParams=pd.read_csv('meanAbsMovementWMLRHCP.csv',index_col=0)


randinds1=np.arange(0,len(subsToUse))
random.shuffle(randinds1)
np.save('Randinds'+iterNum+'.npy',randinds1)
subsToUse=subsToUse[randinds1]
subsToUse = subsToUse[:800]

#### Define HCP Variables

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

# HCP 400 model
pos_fit400,neg_fit400,posedges400,negedges400,_=cpm.train_cpm(HCPDataSample1.T,HCPpmatSample1,corrtype = 'partial', confound=HCPabsmotSample1)
model400edges = np.sum(posedges400)
# PNC Data
PNCmatFile=io.loadmat('PathToPNCWorkingMemoryFCMats') 
PNCData=PNCmatFile['ipmats']
PNCDataRes=np.reshape(PNCData,[268**2,788])

PNCpmatFile=io.loadmat('PathToPNCfIQ')  
PNCpmat=np.squeeze(PNCpmatFile['pmats_pnc'])



### HCP BS Model

bsDictPath='BagDict'+iterNum+'.npy'

if not os.path.isfile(bsDictPath):
    args=(HCPDataSample1,HCPpmatSample1,0,False,400,False,subsToUse[:400],'partial',HCPabsmotSample1)

    bsDict=cpm.run_cpm(args)

    np.save(bsDictPath,bsDict)

else:
    bsDict=np.load(bsDictPath,allow_pickle=True).item()

modelEdgesUnthresh = bsDict['posedges']/100

model = np.mean(np.concatenate([r for r in bsDict['posfits']]),axis=0)
modelEdges=modelEdgesUnthresh > 0


RHCPmodel1_gather=[]
RHCPmodel400_gather=[]
RHCPmodelB_gather=[]
RPNCmodel1_gather=[]
RPNCmodel400_gather=[]
RPNCmodelB_gather=[]


threshList=np.arange(0,1.1,0.1)
threshPerformHCP=np.zeros([len(threshList),100])
threshPerformPNC=np.zeros([len(threshList),100])

threshListLessThan=np.arange(0.1,1.1,0.1)
threshPerformHCPLessThan=np.zeros([len(threshListLessThan),100])
threshPerformPNCLessThan=np.zeros([len(threshListLessThan),100])

model200edges = np.zeros([100,1])


print('Training CPM')
for i in range(0,100):
    print(i)

    randinds=np.arange(0,400)
    random.shuffle(randinds)

    traininds=randinds[:200]
    testinds=randinds[200:400]
    

    trainmats = HCPDataSample1[traininds,:].T
    trainpheno = HCPpmatSample1[traininds]
    trainconfound = HCPabsmotSample1[traininds]
 
    testmatsPNC=PNCDataRes[:,testinds]
    testphenoPNC=PNCpmat[testinds]
    
    testmats=HCPDataSample2[testinds,:].T
    testpheno=HCPpmatSample2[testinds]
    testconfound = HCPabsmotSample2[testinds]


    pos_fit,neg_fit,posedges,negedges,_ = cpm.train_cpm(trainmats,trainpheno,corrtype = 'partial', confound=trainconfound)

    # Test single trained model on HCP
    peHCPmodel1=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2
    neHCPmodel1=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2

    behav_pred_pos_HCPmodel1=pos_fit[0]*peHCPmodel1 + pos_fit[1]
    maskHCPmodel1=~np.isnan(behav_pred_pos_HCPmodel1)
    RposHCPmodel1=np.corrcoef(behav_pred_pos_HCPmodel1[maskHCPmodel1],testpheno[maskHCPmodel1])


    # Test single trained model on PNC
    pePNCmodel1=np.sum(testmatsPNC[posedges.flatten().astype(bool),:], axis=0)/2
    nePNCmodel1=np.sum(testmatsPNC[negedges.flatten().astype(bool),:], axis=0)/2

    behav_pred_pos_PNCmodel1=pos_fit[0]*pePNCmodel1 + pos_fit[1]
    maskPNCmodel1=~np.isnan(behav_pred_pos_PNCmodel1)
    RposPNCmodel1=np.corrcoef(behav_pred_pos_PNCmodel1[maskPNCmodel1],testphenoPNC[maskPNCmodel1])


    # Test single 400 model on HCP

    peHCPmodel400=np.sum(testmats[posedges400.flatten().astype(bool),:], axis=0)/2
    neHCPmodel400=np.sum(testmats[negedges400.flatten().astype(bool),:], axis=0)/2

    behav_pred_pos_HCPmodel400=pos_fit400[0]*peHCPmodel400 + pos_fit400[1]
    maskHCPmodel400=~np.isnan(behav_pred_pos_HCPmodel400)
    RposHCPmodel400=np.corrcoef(behav_pred_pos_HCPmodel400[maskHCPmodel400],testpheno[maskHCPmodel400])

    # Test single 400 model on PNC

    pePNCmodel400=np.sum(testmatsPNC[posedges400.flatten().astype(bool),:], axis=0)/2
    nePNCmodel400=np.sum(testmatsPNC[negedges400.flatten().astype(bool),:], axis=0)/2

    behav_pred_pos_PNCmodel400=pos_fit400[0]*pePNCmodel400 + pos_fit400[1]
    maskPNCmodel400=~np.isnan(behav_pred_pos_PNCmodel400)
    RposPNCmodel400=np.corrcoef(behav_pred_pos_PNCmodel400[maskPNCmodel400],testpheno[maskPNCmodel400])


    # Test bootstrap trained model on HCP
    peHCPmodelBS=np.sum(testmats[modelEdges.flatten().astype(bool),:], axis=0)/2
    
    behav_pred_pos_HCPmodelBS=model[0]*peHCPmodelBS + model[1]
    maskHCPmodelBS=~np.isnan(behav_pred_pos_HCPmodelBS)
    RposHCPmodelBS=np.corrcoef(behav_pred_pos_HCPmodelBS[maskHCPmodelBS],testpheno[maskHCPmodelBS])

    # Test bootstrap trained model on PNC
    pePNCmodelBS=np.sum(testmatsPNC[modelEdges.flatten().astype(bool),:], axis=0)/2

    behav_pred_pos_PNCmodelBS=model[0]*pePNCmodelBS + model[1]
    maskPNCmodelBS=~np.isnan(behav_pred_pos_PNCmodelBS)
    RposPNCmodelBS=np.corrcoef(behav_pred_pos_PNCmodelBS[maskPNCmodelBS],testphenoPNC[maskPNCmodelBS])


    RHCPmodel1_gather.append(RposHCPmodel1[0,1])
    RHCPmodel400_gather.append(RposHCPmodel400[0,1])
    RHCPmodelB_gather.append(RposHCPmodelBS[0,1])
    RPNCmodel1_gather.append(RposPNCmodel1[0,1])
    RPNCmodel400_gather.append(RposPNCmodel400[0,1])
    RPNCmodelB_gather.append(RposPNCmodelBS[0,1])


    
    for threshInd, thresh in enumerate(threshList):
        #pdb.set_trace()
        print("Running thresholded bootstrapped model, > thresh: ",thresh)
        modelEdgesThresh=modelEdgesUnthresh > thresh
        # Test bootstrap trained model on HCP
        peHCPmodelBS=np.sum(testmats[modelEdgesThresh.flatten().astype(bool),:], axis=0)/2
        
        behav_pred_pos_HCPmodelBS=model[0]*peHCPmodelBS + model[1]
        maskHCPmodelBS=~np.isnan(behav_pred_pos_HCPmodelBS)
        threshPerformHCP[threshInd,i]=np.corrcoef(behav_pred_pos_HCPmodelBS[maskHCPmodelBS],testpheno[maskHCPmodelBS])[0,1]

        # Test bootstrap trained model on PNC
        pePNCmodelBS=np.sum(testmatsPNC[modelEdgesThresh.flatten().astype(bool),:], axis=0)/2

        behav_pred_pos_PNCmodelBS=model[0]*pePNCmodelBS + model[1]
        maskPNCmodelBS=~np.isnan(behav_pred_pos_PNCmodelBS)
        threshPerformPNC[threshInd,i]=np.corrcoef(behav_pred_pos_PNCmodelBS[maskPNCmodelBS],testphenoPNC[maskPNCmodelBS])[0,1]


    for threshInd2, thresh2 in enumerate(threshListLessThan):
        print("Running thresholded bootstrapped model, <= thresh: ",thresh2)
        modelEdgesThresh2=(modelEdgesUnthresh > thresh2-0.1) & (modelEdgesUnthresh < thresh2)
        # Test bootstrap trained model on HCP
        peHCPmodelBS2=np.sum(testmats[modelEdgesThresh2.flatten().astype(bool),:], axis=0)/2

        behav_pred_pos_HCPmodelBS2=model[0]*peHCPmodelBS2 + model[1]
        maskHCPmodelBS2=~np.isnan(behav_pred_pos_HCPmodelBS2)
        threshPerformHCPLessThan[threshInd2,i]=np.corrcoef(behav_pred_pos_HCPmodelBS2[maskHCPmodelBS2],testpheno[maskHCPmodelBS2])[0,1]

        # Test bootstrap trained model on PNC
        pePNCmodelBS2=np.sum(testmatsPNC[modelEdgesThresh2.flatten().astype(bool),:], axis=0)/2

        behav_pred_pos_PNCmodelBS2=model[0]*pePNCmodelBS2 + model[1]
        maskPNCmodelBS2=~np.isnan(behav_pred_pos_PNCmodelBS2)
        threshPerformPNCLessThan[threshInd2,i]=np.corrcoef(behav_pred_pos_PNCmodelBS2[maskPNCmodelBS2],testphenoPNC[maskPNCmodelBS2])[0,1]



data=np.stack([RHCPmodel1_gather,RHCPmodel400_gather,RHCPmodelB_gather,RPNCmodel1_gather,RPNCmodel400_gather,RPNCmodelB_gather])


np.save('ModelPerformance'+iterNum+'.npy',data)
np.save('BagModelInfo'+iterNum+'.npy',bsDict)
np.save('ThresholdBaggedModelPerformance'+iterNum+'.npy',{'HCPThresh':threshPerformHCP,'PNCThresh':threshPerformPNC,'HCPThreshLessThan':threshPerformHCPLessThan,'PNCThreshLessThan':threshPerformPNCLessThan})
np.save('S400Parameters'+iterNum+'.npy',{'posfit400':pos_fit400,'negfit400':neg_fit400,'posedges400':posedges400,'negedges400':negedges400})
np.save('S400S200NumEdges'+iterNum+'.npy',{'model400edges':model400edges,'model200edges':model200edges})
