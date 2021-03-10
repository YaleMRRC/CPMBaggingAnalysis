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


import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool



def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * round(y,2))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'





iterResGather=[]
bsEdgesGather=[]
bsEdgesGather=[]
bsEdgesGather=[]

threshPerformGather=[]

cvPerformGather = []

histDenGatherBoot=[]
histAbsGatherBoot=[]
histDenGatherS300=[]
histAbsGatherS300=[]
histDenGatherS200=[]
histAbsGatherS200=[]

edgeDfGather = []

rvalDict={}

for itr in range(1,21):

    rvalDict[itr] = {}

    globalIpdir = '/path/to/iter'+str(itr).zfill(2)
    globalOpdir = '/path/to/figs/'

    edgeCountDf = pd.DataFrame()

    edgemask = np.triu(np.ones([268,268]),k=1).flatten().astype(bool)


    ## Ten fold
    print('Ten Fold')
    tenFoldPath = os.path.join(globalIpdir,'10kCV.npy')
    tenFoldRes = np.load(tenFoldPath, allow_pickle = True).item()


    edgeCountDf['tenFold'] = np.stack(tenFoldRes['res'][0]).reshape([1000,71824])[:,edgemask].sum(axis=1)


    ## Train Only
    print('Train Only')
    trainPath = os.path.join(globalIpdir,'trainOnly.npy')
    trainRes = np.load(trainPath, allow_pickle = True)
    edgeCountDf['trainOnly'] = np.pad(np.expand_dims(trainRes[2][edgemask].sum(),axis=0).astype('object'),[0,999],constant_values=(np.nan,))


    ## Splithalf CV
    print('Splithalf')
    splitHalfPath = os.path.join(globalIpdir,'splithalfCV.npy')
    splitHalfRes = np.load(splitHalfPath, allow_pickle = True).item()

    edgeCountDf['splitHalf'] = np.pad(np.stack(splitHalfRes['res'][0]).reshape([200,71824])[:,edgemask].sum(axis=1).astype('object'),[0,800],constant_values=(np.nan,))

    ## Five fold
    print('Five fold')
    fiveFoldPath = os.path.join(globalIpdir,'5kCV.npy')
    fiveFoldRes = np.load(fiveFoldPath, allow_pickle = True).item()

    edgeCountDf['fiveFold'] = np.pad(np.stack(fiveFoldRes['res'][0]).reshape([500,71824])[:,edgemask].sum(axis=1).astype('object'),[0,500],constant_values=(np.nan,))


    ## LOO
    print('LOO')
    looPath = os.path.join(globalIpdir,'looCV.npy')
    looRes = np.load(looPath, allow_pickle = True).item()

    edgeCountDf['LOO'] = np.pad(np.stack(looRes['res'][0]).reshape([400,71824])[:,edgemask].sum(axis=1).astype('object'),[0,600],constant_values=(np.nan,))







    ## Bootstrap
    print('Bootstrap')
    bootPath = os.path.join(globalIpdir,'bootstrap.npy')
    bootRes = np.load(bootPath, allow_pickle = True).item()

    bootRes = bootRes['res']
    bootedges = np.array(np.array(bootRes[1]).mean(axis=0) > 0)[edgemask]
    bootmodel = np.mean(bootRes[3],axis=0)

    ## Subsample
    print('Subsample 300')
    sub300Path = os.path.join(globalIpdir,'subsample300.npy')
    sub300Res = np.load(sub300Path, allow_pickle = True).item()

    sub300Res = sub300Res['res']
    sub300edges = np.array(np.array(sub300Res[1]).mean(axis=0) > 0)[edgemask]
    sub300model = np.mean(sub300Res[3],axis=0)

    ## Subsample
    print('Subsample 200')
    sub200Path = os.path.join(globalIpdir,'subsample200.npy')
    sub200Res = np.load(sub200Path, allow_pickle = True).item()

    sub200Res = sub200Res['res']
    sub200edges = np.array(np.array(sub200Res[1]).mean(axis=0) > 0)[edgemask]
    sub200model = np.mean(sub200Res[3],axis=0)





    # SplitHalf
    posEdgeMasktwoK = np.concatenate(splitHalfRes['res'][0])
    posFitstwoK = np.concatenate(splitHalfRes['res'][5])

    # 5k
    posEdgeMaskfiveK = np.concatenate(fiveFoldRes['res'][0])
    posFitsfiveK = np.concatenate(fiveFoldRes['res'][5])

    #10K
    posEdgeMasktenK = np.concatenate(tenFoldRes['res'][0])
    posFitstenK = np.concatenate(tenFoldRes['res'][5])

    #loo
    posEdgeMaskloo = np.concatenate(looRes['res'][0])
    posFitsloo = np.concatenate(looRes['res'][5])

    #train only
    trainMod = trainRes[0]
    trainEdges = trainRes[2]


    ## CV Performance evaluated across all folds

    cvPerfDf = pd.DataFrame()

    # 2K
    posBehavRes = np.reshape(splitHalfRes['res'][2],[100,400])
    actBehavRes = np.reshape(splitHalfRes['res'][4],[100,400])

    cvPerf2K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,100)])

    # 5K
    posBehavRes = np.reshape(fiveFoldRes['res'][2],[100,400])
    actBehavRes = np.reshape(fiveFoldRes['res'][4],[100,400])

    cvPerf5K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,100)])

    # 10K
    posBehavRes = np.reshape(tenFoldRes['res'][2],[100,400])
    actBehavRes = np.reshape(tenFoldRes['res'][4],[100,400])

    cvPerf10K = np.array([np.corrcoef(posBehavRes[i,:],actBehavRes[i,:])[0,1] for i in range(0,100)])

    # LOO
    posBehavRes = np.reshape(looRes['res'][2],[400])
    actBehavRes = np.reshape(looRes['res'][4],[400])

    cvPerfloo = np.corrcoef(posBehavRes,actBehavRes)[0,1]


    cvPerfDf['splitHalf'] = cvPerf2K
    cvPerfDf['fiveFold'] = cvPerf5K
    cvPerfDf['tenFold'] = cvPerf10K
    cvPerfDf['LOO'] = np.pad(np.expand_dims(cvPerfloo,axis=0).astype('object'),[0,99],constant_values=(np.nan,))

    cvPerformGather.append(cvPerfDf)

    bootedgesAv = np.array(bootRes[1])[:,edgemask].mean(axis=0)
    sub300edgesAv = np.array(sub300Res[1])[:,edgemask].mean(axis=0)
    sub200edgesAv = np.array(sub200Res[1])[:,edgemask].mean(axis=0)
    posEdgeMasktwoKAv = np.concatenate(splitHalfRes['res'][0])[:,edgemask].mean(axis=0)
    posEdgeMaskfiveKAv = np.concatenate(fiveFoldRes['res'][0])[:,edgemask].mean(axis=0)
    posEdgeMasktenKAv = np.concatenate(tenFoldRes['res'][0])[:,edgemask].mean(axis=0)
    posEdgeMasklooAv = np.concatenate(looRes['res'][0])[:,edgemask].mean(axis=0)


    for thresh in np.arange(0,1,0.1):
        
        if thresh == 0:
            bootNum = np.array(bootedgesAv > thresh).sum()
            s300Num = np.array(sub300edgesAv > thresh).sum()
            s200Num = np.array(sub200edgesAv > thresh).sum()
 
            edgeCountDf['boot>'+str(round(thresh,1))] = np.pad(np.expand_dims(bootNum,axis=0).astype('object'),[0,999],constant_values=(np.nan,))
            edgeCountDf['sub300>'+str(round(thresh,1))] = np.pad(np.expand_dims(s300Num,axis=0).astype('object'),[0,999],constant_values=(np.nan,))
            edgeCountDf['sub200>'+str(round(thresh,1))] = np.pad(np.expand_dims(s200Num,axis=0).astype('object'),[0,999],constant_values=(np.nan,))

        else:
            bootNum = np.array(bootedgesAv >= thresh).sum()
            s300Num = np.array(sub300edgesAv >= thresh).sum()
            s200Num = np.array(sub200edgesAv >= thresh).sum()
 
            edgeCountDf['boot>='+str(round(thresh,1))] = np.pad(np.expand_dims(bootNum,axis=0).astype('object'),[0,999],constant_values=(np.nan,))
            edgeCountDf['sub300>='+str(round(thresh,1))] = np.pad(np.expand_dims(s300Num,axis=0).astype('object'),[0,999],constant_values=(np.nan,))
            edgeCountDf['sub200>='+str(round(thresh,1))] = np.pad(np.expand_dims(s200Num,axis=0).astype('object'),[0,999],constant_values=(np.nan,))






    #edgeCountDf=edgeCountDf.unstack().reset_index()
    #edgeCountDf=edgeCountDf.rename({0:'NumberOfEdges'},axis=1)   
    #edgeCountDf.dropna(inplace=True)
    #edgeCountDf.reset_index(inplace=True)
    dfLen = edgeCountDf.shape[0]
    edgeCountDf['sampleNum'] = np.repeat([itr],dfLen)
    edgeDfGather.append(edgeCountDf)

    #### Creating DataFrames



    threshResPath = os.path.join(globalIpdir,'threshRes.npy')
    threshRes = np.load(threshResPath, allow_pickle = True).item()


    threshDfGather = []
    for k1 in threshRes.keys():
        for k2 in threshRes[k1].keys():
            for k3 in threshRes[k1][k2].keys():
                for k4 in threshRes[k1][k2][k3].keys():
                    val = threshRes[k1][k2][k3][k4]
                    row = np.vstack(np.array([k1,k2,k3,k4,val,0])).T
                    miniDf = pd.DataFrame(row,columns = ['testSize','iter','thresh','modelTestSample','pearsonsR','modelNum'])
                    threshDfGather.append(miniDf)


    threshDf = pd.concat(threshDfGather)
    threshDf = threshDf.reset_index(drop=True)
    threshDf['modelType'] = threshDf.modelTestSample.str.replace('PNC','').str.replace('HCP','')
    threshDf['testSample'] = threshDf.testSize.str[:3]
    threshDf.testSize = threshDf.testSize.str[3:]
    threshDf.pearsonsR = threshDf.pearsonsR.astype('float')
    dfLen = threshDf.shape[0]
    threshDf['sampleNum'] = np.repeat([itr],dfLen)


    iterResPath = os.path.join(globalIpdir,'iterRes.npy')
    iterRes = np.load(iterResPath, allow_pickle = True).item()


    iterDfGather = []
    for k1 in iterRes.keys():
        for k2 in iterRes[k1].keys():
            for k3 in iterRes[k1][k2].keys():
                val = iterRes[k1][k2][k3]
                if type(val) == np.float64:
                    rows = np.vstack(np.array([k1,k2,k3,val,0])).T
                else:
                    nrows = val.shape[0]
                    modelNum = np.arange(0,nrows)
                    rows = np.append(np.tile([k1,k2,k3],nrows).reshape(nrows,3),np.vstack(val),axis=1)
                    rows = np.append(rows,np.vstack(modelNum),axis=1)
                miniDf = pd.DataFrame(rows,columns = ['testSize','iter','modelTestSample','pearsonsR','modelNum'])
                iterDfGather.append(miniDf)

    iterDf = pd.concat(iterDfGather)
    iterDf = iterDf.reset_index(drop=True)

    iterDf['modelType'] = iterDf.modelTestSample.str.replace('PNC','').str.replace('HCP','')
    iterDf['testSample'] = iterDf.testSize.str[:3]
    iterDf.testSize = iterDf.testSize.str[3:]
    iterDf.pearsonsR = iterDf.pearsonsR.astype('float')
    dfLen = iterDf.shape[0]
    iterDf['sampleNum'] = np.repeat([itr],dfLen)



    iterResGather.append(iterDf)
    threshPerformGather.append(threshDf)


    # Aggregate performance of all models within and out of sample
    
    fig, ax = plt.subplots(figsize=[12,8],nrows=1,ncols=2)

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    sns.violinplot(data=iterDf[iterDf.testSample == 'hcp'],y="pearsonsR",x='modelType',inner='quartile',hue='testSize',ax = ax[0]) 

    sns.despine(left=True, bottom=False)
    ax[0].set_title('Performance of single and bootstrapped models within sample', fontsize=14)
    ax[0].yaxis.grid(True)
    ax[0].xaxis.grid(False)
    #plt.ylim([-0.2,0.65])
    #plt.xlim([-0.4,1.4])


    sns.violinplot(data=iterDf[iterDf.testSample == 'pnc'],y="pearsonsR",x='modelType',inner='quartile',hue='testSize',ax = ax[1]) 

    sns.despine(left=True, bottom=False)
    ax[1].set_title('Performance of single and bootstrapped models out of sample', fontsize=14)
    ax[1].yaxis.grid(True)
    ax[1].xaxis.grid(False)
    #plt.ylim([-0.2,0.65])
    #plt.xlim([-0.4,1.4])


    plt.tight_layout()

    plt.savefig(os.path.join(globalOpdir,'BootstrapComparisonViolin'+str(itr)+'.png'))
    plt.close()


    #plt.clf()
    plt.close('all')




    
    def histPlot(ipArr,opname):
        ### Distribution of feature inclusion 

        # from https://matplotlib.org/2.0.2/examples/pylab_examples/broken_axis.html
        plt.figure(figsize=[8,6])
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        sns.set(style="whitegrid", palette="bright", color_codes=True)
        # plot the same data on both axes
        x,y,rects=ax.hist(ipArr[ipArr > 0],range=(0,1))

        ax2.hist(ipArr[ipArr > 0],range=(0,1))
        ax.set_title('Distribution of feature occurence across bootstraps', fontsize=14)
        # zoom-in / limit the view to different portions of the data
        ax.set_ylim(10000, 15000)  # outliers only
        ax2.set_ylim(0, 1600)  # most of the data

        # hide the spines between ax and ax2
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['top'].set_visible(False)
        #ax2.spines['top'].set_visible(False)
        #ax.xaxis.tick_top()
        ax.tick_params(bottom=False)  # don't put tick labels at the top
        #ax2.xaxis.tick_bottom()

        # This looks pretty good, and was fairly painless, but you can get that
        # cut-out diagonal lines look with just a bit more work. The important
        # thing to know here is that in axes coordinates, which are always
        # between 0-1, spine endpoints are at these locations (0,0), (0,1),
        # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
        # appropriate corners of each of our axes, and so long as we use the
        # right transform and disable clipping.

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        #ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        # What's cool about this is that now if we vary the distance between
        # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
        # the diagonal lines will move accordingly, and stay right at the tips
        # of the spines they are 'breaking'
        plt.xlabel('Frequency of occurence', fontsize=12)
        plt.ylabel('Number of edges', fontsize=12)

        sns.despine(ax=ax,left=False,bottom=True)
        sns.despine(ax=ax2,left=False, bottom=False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax2.xaxis.grid(False)
        ax2.yaxis.grid(False)
        plt.tight_layout()


        plt.savefig(opname)
        plt.close()


        plt.clf()

        return x

    x = histPlot(bootedgesAv,os.path.join(globalOpdir,'BootstrapEdgeDistribution'+str(itr)+'.png'))
    histAbsGatherBoot.append(x)
    x = histPlot(sub300edgesAv,os.path.join(globalOpdir,'Sub300EdgeDistribution'+str(itr)+'.png'))
    histAbsGatherS300.append(x)
    x = histPlot(sub200edgesAv,os.path.join(globalOpdir,'Sub200EdgeDistribution'+str(itr)+'.png'))
    histAbsGatherS200.append(x)

    #################################################
    ############ BS model thresholding ##############
    #################################################





    for testSize in threshDf[threshDf.testSample == 'hcp'].testSize.unique():


        tempThreshDf = threshDf[threshDf.testSample == 'hcp']
        tempThreshDf['tempInd'] = list(map(lambda x: '_'.join(list(x)),tempThreshDf[['iter','modelTestSample']].values))

        bootvals = tempThreshDf[(tempThreshDf.modelTestSample == 'bootHCP') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values
        sub300vals = tempThreshDf[(tempThreshDf.modelTestSample == 'sub300HCP') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values
        sub200vals = tempThreshDf[(tempThreshDf.modelTestSample == 'sub200HCP') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values

        f = plt.figure(figsize=[10,12])
        sns.set_style('white')
        ax = f.add_subplot(311)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax.boxplot(bootvals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
        ax.set_ylim(0,0.6)
        ax.set_ylabel('Performance (R)')
        #ax.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax2 = ax.twinx()
        x,y,rects=ax2.hist(bootedgesAv[bootedgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)
        histDenGatherBoot.append(x)
        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax2.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax2.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax2.yaxis.set_major_formatter(formatter)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax.set_xlim(-0.05,1.05)

        ax.set_xlabel('Percentage of boostraps features occured in')
        ax.set_title('Bootstrap model performance with feature thresholding HCP->HCP')


        ### Bootstrap model threshold model in HCP

        ax3 = f.add_subplot(312)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax3.boxplot(sub300vals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
        ax3.set_ylim(0,0.6)
        ax3.set_ylabel('Performance (R)')

        ax4 = ax3.twinx()


        x,y,rects=ax4.hist(sub300edgesAv[sub300edgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)
        histDenGatherS300.append(x)
        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax4.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax4.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax4.yaxis.set_major_formatter(formatter)



        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax3.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax3.set_xlabel('Percentage of boostraps features occured in')
        ax3.set_title('Resample 300 model performance with feature thresholding HCP->HCP')






        ax5 = f.add_subplot(313)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels3=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax5.boxplot(sub200vals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
        ax5.set_ylim(0,0.6)
        ax5.set_ylabel('Performance (R)')

        ax6 = ax5.twinx()


        x,y,rects=ax5.hist(sub200edgesAv[sub200edgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)
        histDenGatherS200.append(x)
        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax6.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax6.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax6.yaxis.set_major_formatter(formatter)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax3.xaxis.grid(False)
        ax3.yaxis.grid(True)
        ax5.xaxis.grid(False)
        ax5.yaxis.grid(True)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax5.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax5.set_xlabel('Percentage of boostraps features occured in')
        ax5.set_title('Resample 200 model performance with feature thresholding HCP->HCP')




        plt.tight_layout()
        plt.savefig(os.path.join(globalOpdir,'HCPThreshPerform_testSize'+testSize+'_'+str(itr)+'.png'))
        plt.close()



    for testSize in threshDf[threshDf.testSample == 'pnc'].testSize.unique():


        tempThreshDf = threshDf[threshDf.testSample == 'pnc']
        tempThreshDf['tempInd'] = list(map(lambda x: '_'.join(list(x)),tempThreshDf[['iter','modelTestSample']].values))

        bootvals = tempThreshDf[(tempThreshDf.modelTestSample == 'bootPNC') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values
        sub300vals = tempThreshDf[(tempThreshDf.modelTestSample == 'sub300PNC') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values
        sub200vals = tempThreshDf[(tempThreshDf.modelTestSample == 'sub200PNC') & (tempThreshDf.testSize == testSize)][['thresh','pearsonsR','tempInd']].pivot(columns = 'thresh',index='tempInd').values


        f = plt.figure(figsize=[10,12])
        sns.set_style('white')
        ax = f.add_subplot(311)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax.boxplot(bootvals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
        ax.set_ylim(0,0.6)
        ax.set_ylabel('Performance (R)')
        #ax.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax2 = ax.twinx()
        x,y,rects=ax2.hist(bootedgesAv[bootedgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)

        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax2.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax2.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax2.yaxis.set_major_formatter(formatter)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax.set_xlim(-0.05,1.05)

        ax.set_xlabel('Percentage of boostraps features occured in')
        ax.set_title('Bootstrap model performance with feature thresholding HCP->PNC')


        ### Bootstrap model threshold model in HCP

        ax3 = f.add_subplot(312)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax3.boxplot(sub300vals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
        ax3.set_ylim(0,0.6)
        ax3.set_ylabel('Performance (R)')

        ax4 = ax3.twinx()


        x,y,rects=ax4.hist(sub300edgesAv[sub300edgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)

        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax4.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax4.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax4.yaxis.set_major_formatter(formatter)



        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax3.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax3.set_xlabel('Percentage of boostraps features occured in')
        ax3.set_title('Resample 300 model performance with feature thresholding HCP->PNC')






        ax5 = f.add_subplot(313)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels3=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax5.boxplot(sub200vals,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
        ax5.set_ylim(0,0.6)
        ax5.set_ylabel('Performance (R)')

        ax6 = ax5.twinx()


        x,y,rects=ax5.hist(sub200edgesAv[sub200edgesAv > 0],10,range=(0,1),cumulative=-1,density=True,alpha=0.3)

        for i,rect in enumerate(rects):
            txt="{0:.1%}".format(rect.get_height())
            ax6.text(rect.get_x()+0.05, rect.get_height(),txt, ha='center', va='bottom',alpha=0.5)
        ax6.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax6.yaxis.set_major_formatter(formatter)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax3.xaxis.grid(False)
        ax3.yaxis.grid(True)
        ax5.xaxis.grid(False)
        ax5.yaxis.grid(True)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax5.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax5.set_xlabel('Percentage of boostraps features occured in')
        ax5.set_title('Resample 200 model performance with feature thresholding HCP->PNC')




        plt.tight_layout()
        plt.savefig(os.path.join(globalOpdir,'PNCThreshPerform_testSize'+testSize+'_'+str(itr)+'.png'))
        plt.close()



    ## Feature inclusion plot

    fig, ax = plt.subplots(figsize=[8,6])
    sns.set(style="white", palette="bright", color_codes=True)
    featRvalMean=np.mean(bootRes[5],axis=0) 
    posedges=np.stack(bootRes[1]).mean(axis=0)
    negedges=np.stack(bootRes[2]).mean(axis=0)
    sns.set_style("whitegrid")
    plt.scatter(featRvalMean[(featRvalMean > 0) & (posedges> 0)],posedges[(featRvalMean > 0) & (posedges > 0)])
    plt.scatter(featRvalMean[(featRvalMean < 0) & (negedges > 0)],negedges[(featRvalMean < 0) & (negedges > 0)])
    plt.title('Bagged Model Rvals')
    plt.xlabel('Rval for feature vs behavior')
    plt.ylabel('Percentage of bootstraps feature occured in')
    sns.despine(right=True)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.ylim([0,1.1])
    plt.xlim([-0.4,0.4])
    plt.savefig(os.path.join(globalOpdir,'featureRvalBootstrap'+str(itr)+'.png'))
    plt.close('all')



    fig, ax = plt.subplots(figsize=[8,6])
    sns.set(style="white", palette="bright", color_codes=True)
    featRvalMean=np.mean(sub300Res[5],axis=0) 
    posedges=np.stack(sub300Res[1]).mean(axis=0)
    negedges=np.stack(sub300Res[2]).mean(axis=0)
    sns.set_style("whitegrid")
    plt.scatter(featRvalMean[(featRvalMean > 0) & (posedges> 0)],posedges[(featRvalMean > 0) & (posedges > 0)])
    plt.scatter(featRvalMean[(featRvalMean < 0) & (negedges > 0)],negedges[(featRvalMean < 0) & (negedges > 0)])
    plt.title('S300 Model Rvals')
    plt.xlabel('Rval for feature vs behavior')
    plt.ylabel('Percentage of resamples feature occured in')
    sns.despine(right=True)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.ylim([0,1.1])
    plt.xlim([-0.4,0.4])
    plt.savefig(os.path.join(globalOpdir,'featureRvalS300'+str(itr)+'.png'))
    plt.close('all')



    fig, ax = plt.subplots(figsize=[8,6])
    sns.set(style="white", palette="bright", color_codes=True)
    featRvalMean=np.mean(sub200Res[5],axis=0) 
    posedges=np.stack(sub200Res[1]).mean(axis=0)
    negedges=np.stack(sub200Res[2]).mean(axis=0)
    sns.set_style("whitegrid")
    plt.scatter(featRvalMean[(featRvalMean > 0) & (posedges> 0)],posedges[(featRvalMean > 0) & (posedges > 0)])
    plt.scatter(featRvalMean[(featRvalMean < 0) & (negedges > 0)],negedges[(featRvalMean < 0) & (negedges > 0)])
    plt.title('S200 Model Rvals')
    plt.xlabel('Rval for feature vs behavior')
    plt.ylabel('Percentage of resamples feature occured in')
    sns.despine(right=True)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.ylim([0,1.1])
    plt.xlim([-0.4,0.4])
    plt.savefig(os.path.join(globalOpdir,'featureRvalS200'+str(itr)+'.png'))
    plt.close('all')




    fig, ax = plt.subplots(figsize=[8,6])
    sns.set(style="white", palette="bright", color_codes=True)

    sns.set_style("whitegrid")
    plt.scatter(np.mean(bootRes[5],axis=0),np.stack(bootRes[1]).mean(axis=0)+np.stack(bootRes[2]).mean(axis=0))
    plt.scatter(np.mean(sub300Res[5],axis=0),np.stack(sub300Res[1]).mean(axis=0)+np.stack(sub300Res[2]).mean(axis=0))
    plt.scatter(np.mean(sub200Res[5],axis=0),np.stack(sub200Res[1]).mean(axis=0)+np.stack(sub200Res[2]).mean(axis=0))
    plt.title('Bagged Model Rvals')
    plt.xlabel('Rval for feature vs behavior')
    plt.ylabel('Percentage of bootstraps feature occured in')
    sns.despine(right=True)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.ylim([0,1.1])
    plt.xlim([-0.4,0.4])
    plt.savefig(os.path.join(globalOpdir,'featureRval3Model'+str(itr)+'.png'))
    plt.close('all')


    # Save feature selection stuff

    rvalDict[itr]['bootRvals'] = np.mean(bootRes[5],axis=0)
    rvalDict[itr]['sub300Rvals'] = np.mean(sub300Res[5],axis=0)
    rvalDict[itr]['sub200Rvals'] = np.mean(sub200Res[5],axis=0)

    rvalDict[itr]['bootEdgeCount'] = np.stack(bootRes[1]).mean(axis=0)+np.stack(bootRes[2]).mean(axis=0)
    rvalDict[itr]['sub300EdgeCount'] = np.stack(sub300Res[1]).mean(axis=0)+np.stack(sub300Res[2]).mean(axis=0)
    rvalDict[itr]['sub200EdgeCount'] = np.stack(sub200Res[1]).mean(axis=0)+np.stack(sub200Res[2]).mean(axis=0)




################################# End of big loop







allIterResDf =  pd.concat(iterResGather)


allIterResDf['R Squared']=allIterResDf['pearsonsR']**2





#### Pairwise performance differences


newind = list(allIterResDf.modelType.unique())

mdlList = np.array([[n1,n2] for n1 in newind for n2 in newind])

mdlListRes = mdlList.reshape([8,8,2])

mdlListKeep = mdlListRes[~np.triu(np.ones([8,8])).astype(bool),:]
mdlMask = np.array(list(map(lambda x : x.split('_')[0], mdlListKeep[:,0]))) != np.array(list(map(lambda x : x.split('_')[0], mdlListKeep[:,1])))
mdlListUnq = mdlListKeep[mdlMask,:]

del mdlMask,mdlListKeep,mdlListRes,mdlList,newind


mdlListUnq = [['sub300','LOO'],['sub200','LOO']]


#timeStart = datetime.datetime.now()

#bigValsGather = []
arrDfGather = []


iterResDf2 = allIterResDf[~((allIterResDf.testSample == 'pnc') & (allIterResDf.testSize == '400'))]

iterResDf2.testSize.replace({'400':'All','787':'All'},inplace=True)




for testSize in ['200']:
    for testSample in ['hcp','pnc']:
        
        allIterResDfSizeSample = iterResDf2[(iterResDf2.testSize == testSize) & (iterResDf2.testSample == testSample)]


        for mdlCombo in mdlListUnq:
            mdlCombo = mdlCombo
            print(testSize,testSample,mdlCombo)
            allIterResDfModel = allIterResDfSizeSample[(allIterResDfSizeSample.modelType == mdlCombo[0]) | (allIterResDfSizeSample.modelType == mdlCombo[1])]

            allIterPivot = allIterResDfModel.pivot(columns = ['iter','testSample','sampleNum'],index=['modelType','modelNum'],values='R Squared')
            allIterPivot = allIterPivot.astype(np.float32)


            nrows=allIterPivot.shape[0]

            newind = list(map(lambda x : '_'.join(x),allIterPivot.index.values))

            mdlList = np.array([[n1,n2] for n1 in newind for n2 in newind])

            mdlListRes = mdlList.reshape([nrows,nrows,2])

            mdlListKeep = mdlListRes[np.triu(np.ones([nrows,nrows]),k=1).astype(bool),:]
            mdlMask = np.array(list(map(lambda x : x.split('_')[0], mdlListKeep[:,0]))) != np.array(list(map(lambda x : x.split('_')[0], mdlListKeep[:,1])))
            mdlListKeep = mdlListKeep[mdlMask,:]


            arrAgg = []


            comboDf = pd.DataFrame(allIterPivot.index.values + allIterPivot.index.values[:,None])
            mask = np.triu(np.ones(comboDf.shape),k=1).astype(bool)
            m1 = comboDf.mask(~mask).values[0,-1][0]
            m2 = comboDf.mask(~mask).values[0,-1][2]


            for col in allIterPivot.columns:
                #print(col)
                arr = allIterPivot[col].values - allIterPivot[col].values[:,None]
                arr = arr[np.triu(np.ones(arr.shape),k=1).astype(bool)]
                arr = arr[mdlMask]
                arrAgg.append(arr)

            arrCat = np.concatenate(arrAgg)

            narrrows = arrCat.shape[0]

            arrDf = pd.DataFrame(arrCat,columns=['modelPerfDiff'])

            arrDf['models'] = np.repeat(m1+'_'+m2,narrrows)
            arrDf['sample'] = np.repeat(testSample,narrrows)

            arrDfGather.append(arrDf)
            
            #bigValsGather.append([testSize,testSample,m1,m2,np.sum(arrCat > 0),np.sum(arrCat < 0),np.sum(arrCat == 0)])


#exactTestDf = pd.DataFrame(bigValsGather,columns =['testSize','testSample','model1','model2','model1Better','model2Better','tie'])
#exactTestDf.to_csv('path/to/exacTestStuff.csv')

#timeEnd = datetime.datetime.now()

arrDfBig = pd.concat(arrDfGather)
plt.close()

sns.set(style="whitegrid", palette="bright", color_codes=True,font_scale=1.2)

g = sns.FacetGrid(data=arrDfBig,col="models",row="sample",height=6)
g.map(sns.kdeplot,"modelPerfDiff",fill=True,linewidth=0,common_norm=False)

#g.map(sns.displot,"modelPerfDiff",kind='kde',fill=True,linewidth=0,common_norm=False)


axtitles = ['Subsample 300 > LOO (Within Sample, HCP)',
'Subsample 200 > LOO (Within Sample, HCP)',
'Subsample 300 > LOO (Out of Sample, PNC)',
'Subsample 200 > LOO (Out of Sample, PNC)']


for i,a in enumerate(g.axes.flatten()):
    
    #ax.set_ylim([-0.05,0.4])
    #ax.set_xlim([-1,16])
    a.xaxis.grid(False)
    a.yaxis.grid(False)
    a.set_xticks([0], minor=False)

    a.set_title(axtitles[i])
    a.xaxis.grid(True,which = 'Major')

    if i > 1:
        a.set_xticks([-0.10,-0.05,0.05,0.10], minor=True)
        a.tick_params(axis='x', which='minor', bottom=True,labelsize='small',labelbottom=True)
        a.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
        #plt.setp(a.get_xticklabels(minor=True), visible=True)


#plt.ylabel('Density')
sns.despine(left=False, bottom = False)

g.set_ylabels('Density')
g.set_xlabels('Difference in Model Performance')

g.fig.tight_layout()



plt.savefig(os.path.join(globalOpdir,'999modelPerfCompareDiff.png'))
plt.close()








#exactTestDf = pd.read_csv('path/to/exacTestStuff.csv')
#stats.fisher_exact([[431802,368196],[633125,166875]]) # sub300 loo comp
#stats.fisher_exact([[436709,363291],[579617,220383]]) # sub200 loo comp



# Mean Perf
# allIterResDf.drop(['index','pearsonsR','modelTestSample','iter','sampleNum'],axis=1).groupby(['modelType','testSample']).mean()

# reshapedDf = allIterResDf.drop(['index','pearsonsR','modelTestSample','iter','sampleNum'],axis=1).reset_index().pivot(columns = ['modelType','testSample'],values='R Squared')
# allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')
# allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')['LOO'].mean()

allThreshResDf =  pd.concat(threshPerformGather)



################################ Summary Tables

allIterResDf.drop(['pearsonsR','sampleNum','modelTestSample'],axis=1).groupby(['modelType','testSample','testSize']).mean()
allIterResDf.replace({'sub300Resample':'Subsample 300','sub200Resample': 'Subsample 200'},inplace=True)
reshapePerfDf = allIterResDf.drop(['pearsonsR','sampleNum','modelTestSample'],axis=1).reset_index().pivot(columns = ['modelType','testSample','testSize'],values='R Squared')
meanPerfTable = reshapePerfDf.mean().sort_values(ascending=False)




iterResPlotDf = allIterResDf.replace({'boot':'Bagged','sub300': 'Subsample 300','sub200':'Subsample 200','2K':'Split Half','5K':'Five Fold','10K':'Ten Fold','Train':'Train Only','pnc':'PNC','hcp':'HCP'})

iterResPlotDf2 = iterResPlotDf[~((iterResPlotDf.testSample == 'PNC') & (iterResPlotDf.testSize == '400'))]



iterResPlotDf2.testSize.replace({'400':'All','787':'All'},inplace=True)







################################ Figures of All


for testSampleSize in ['200','300','All']:
    #f = plt.figure(figsize=[24,10])

    #gs  = matplotlib.gridspec.GridSpec(1, 1, right=0.77)
    #ax=plt.subplot(gs[0])



    ### Bootstrap model performance all edges, Single model vs BS model, test sample size 200
    fig, ax = plt.subplots(figsize=[8,6])


    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    d = sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='testSample',hue='modelType',order = ['HCP','PNC'])



    cols=np.zeros([1,20])+1
    cols[:,10:] = 2

    d.pcolorfast((-1,2), (-0.3,0.7),cols,cmap='brg', alpha=0.1)


    fracOff=1/3.72

    #plt.plot([-fracOff, -fracOff, 0, 0], [0.6, 0.62, 0.62, 0.6], lw=1.5, c='k')
    #plt.text(-fracOff/2, 0.62, "*", ha='center', va='bottom', color='k')
    #plt.plot([fracOff, fracOff, 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text(0, 0.61, "*", ha='center', va='bottom', color='k',weight="bold")

    #plt.plot([1-fracOff, 1-fracOff, 1+fracOff, 1+fracOff], [0.45, 0.47, 0.47, 0.45], lw=1.5, c='k')
    #plt.text(1+fracOff, 0.42, "*", ha='center', va='bottom', color='k',weight="bold")
    #plt.plot([(1/3.8), (1/3.8), 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text((1/7.6), 0.59, "*", ha='center', va='bottom', color='k')


    sns.despine(left=True, bottom=True)
    #plt.set_title('Performance of all models within and out of sample', fontsize=14)
    #ax.set_title('Within Sample \t\t\t\t\t\t\t\t Out of Sample \t\t'.expandtabs(), fontsize=40)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.ylim([-0.05,0.4])
    plt.xlim([-0.5,1.5])
    plt.ylabel('Performance (R Squared)')
    plt.xticks([0,1],['Within Sample (HCP)','Out of Sample (PNC)'])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'3BootstrapComparisonBoxplotAllTestSample'+testSampleSize+'.png'))
    plt.close()







fig, ax = plt.subplots(figsize=[20,10])


sns.set(style="whitegrid", palette="pastel", color_codes=True)


d = sns.catplot(x="modelType", y="R Squared",hue="testSize", col="testSample",data=iterResPlotDf2, kind="box",col_order = ['HCP','PNC'],legend = False)

d.set_xticklabels(rotation=30)


sns.despine(left=True, bottom=True)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
plt.ylim([-0.05,0.4])
#plt.xlim([-0.5,1.5])
#plt.ylabel('Performance (R Squared)')
plt.legend(loc='upper right',title = 'Test Sample Size')
plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'3BootstrapComparisonBoxplotAllTestSample3TestSize.png'))
plt.close()






### Three tier plot
cvPerformAll = pd.concat(cvPerformGather)
cvPerformAllStack = cvPerformAll.stack().reset_index()
cvPerformAllStack.columns = ['index','cvType','Pearsons R']
cvPerformAllStack['R Squared'] = cvPerformAllStack['Pearsons R']**2

## R squared violinplot
plt.figure(figsize=[12,10])
plt.subplot(3,1,1)
plt.title('CV Performance')
sns.boxplot(data = cvPerformAllStack, x = 'cvType', y = 'R Squared',color='white',order = ['splitHalf','fiveFold', 'tenFold', 'LOO'])
plt.xticks(range(0,8),['Split Half','Five Fold','Ten Fold','Leave One Out','','','',''])
plt.ylim([-0.1,0.3])
plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

plt.subplot(3,1,2)
plt.title('Performance on left out HCP')
sns.boxplot(data = iterResPlotDf[iterResPlotDf.testSample == 'HCP'], x = 'modelType', y = 'R Squared',hue = 'testSize',color='white',order = ['Split Half','Five Fold', 'Ten Fold', 'LOO', 'Train Only','Subsample 200','Subsample 300','Bagged'])
#plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
plt.ylim([-0.1,0.3])
plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

plt.subplot(3,1,3)
plt.title('Performance on PNC')
sns.boxplot(data = iterResPlotDf[iterResPlotDf.testSample == 'PNC'], x = 'modelType', y = 'R Squared',color='white',hue = 'testSize',order = ['Split Half','Five Fold', 'Ten Fold', 'LOO', 'Train Only','Subsample 200','Subsample 300','Bagged'])
#plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bagged','Train Only'])
plt.ylim([-0.1,0.3])
plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'allPerfIterSqViolin.png'))

plt.close()
plt.clf()




###### All Histogram



def makeHistAll(allAbshist,allAbshistMean,allAbshistStd,opname,cutnums,titl):

    #f = plt.figure(figsize=[12,12])
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    sns.set(style="whitegrid", palette="bright", color_codes=True)

    #ax.set_title(titl, fontsize=40)

    labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']

    ax.bar(np.arange(0.05,1.05,0.1), allAbshistMean, yerr=allAbshistStd,alpha=1,width=0.1) 
    ax2.bar(np.arange(0.05,1.05,0.1), allAbshistMean, yerr=allAbshistStd,alpha=1,width=0.1) 

    ax.set_ylim(cutnums[1], cutnums[0])  # outliers only
    ax2.set_ylim(cutnums[3], cutnums[2])  # most of the data

    ax.tick_params(bottom=False)  # don't put tick labels at the top
    ax.set_xlim(-0.05,1.05)


    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)


    sns.despine(ax=ax,left=False,bottom=True)
    sns.despine(ax=ax2,left=False, bottom=False)

    ax2.set_xlabel('Threshold',fontsize=16)
    ax2.set_ylabel('Number of features', fontsize=16)
    ax.set_title(titl, fontsize=20)
    plt.tight_layout()
    plt.savefig(opname)
    plt.close()

#ax2.xaxis.tick_bottom()

allAbshist=np.stack(histAbsGatherBoot)
allAbshistMean=np.mean(allAbshist,axis=0)
allAbshistStd=np.std(allAbshist,axis=0)
opname1=os.path.join(globalOpdir,'histogramAllBoot.png')

makeHistAll(allAbshist,allAbshistMean,allAbshistStd,opname1,[12500,10000,3000,0],'Bagged Model')


allAbshist=np.stack(histAbsGatherS300)
allAbshistMean=np.mean(allAbshist,axis=0)
allAbshistStd=np.std(allAbshist,axis=0)
opname2=os.path.join(globalOpdir,'histogramAllS300.png')

makeHistAll(allAbshist,allAbshistMean,allAbshistStd,opname2,[3000,2000,800,0],'Subsample 300 Model')

allAbshist=np.stack(histAbsGatherS200)
allAbshistMean=np.mean(allAbshist,axis=0)
allAbshistStd=np.std(allAbshist,axis=0)
opname3=os.path.join(globalOpdir,'histogramAllS200.png')

makeHistAll(allAbshist,allAbshistMean,allAbshistStd,opname3,[7000,900,750,0],'Subsample 200 Model')

os.system('montage '+opname1+' '+opname2+' '+opname3+' -geometry +3+1 '+os.path.join(globalOpdir,'4histogramAllMontage.png'))



###### Performance by split grid form



#f = plt.figure()

d = sns.catplot(x="modelType", y="R Squared",hue="sampleNum", col="testSample",row = 'testSize',data=iterResPlotDf2, kind="box",col_order = ['HCP','PNC'],legend = False,height=5,aspect=2,sharex=False)
#sns.despine(offset=15)
d.set_xticklabels(rotation=15)



plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'2PerformanceComparisonSplitGrid.png'))
plt.close()



###### Figure performance by split, each test size separate
for testSize in ['200','300','All']:
    f = plt.figure(figsize=[40,18])
    gs  = matplotlib.gridspec.GridSpec(1, 1, right=0.85)

    #ax = f.add_subplot(211)
    ax=plt.subplot(gs[0])
    order = ['bootHCP','sub300HCP','sub200HCP','HCP2K','HCP5K','HCP10K','HCPLOO','HCPTrain','bootPNC','sub300PNC','sub200PNC','PNC2K','PNC5K','PNC10K','PNCLOO','PNCTrain']
    d=sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSize],y="R Squared",x='modelTestSample',hue='sampleNum',ax=ax,order=order)
    cols=np.zeros([1,160])+1
    cols[:,80:] = 2


    d.pcolorfast((-1,16), (-0.05,0.4),cols,cmap='brg', alpha=0.1)
    sns.despine(offset=15)
    #ax3.legend(bbox_to_anchor=(1.8, 1.5), loc='upper right')
    #ax3.legend(bbox_to_anchor=(1.3, 1.5), loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='center right',bbox_to_anchor=(0.9, 0.55),fontsize=25,title = 'HCP Split',title_fontsize=25)
    ax.get_legend().remove()
    ax.set_ylim([-0.05,0.4])
    ax.set_xlim([-1,16])

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.title('Within Sample \t\t\t\t\t\t\t\t\t\t\t Out of Sample \t\t'.expandtabs(), fontsize=40)
    plt.xticks(ticks = range(0,16), labels = ['Bagged', 'Subsample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only','Bagged', 'Subample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only'],rotation=25,fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Performance (R Squared)', fontsize=30)
    plt.xlabel('')
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'2PerformanceComparisonSplit'+testSize+'.png'))
    plt.close()





######################### Overlay figures 2 and 3 test


###### Figure performance by split, each test size separate
for testSize in ['200','300','All']:
    fig, ax = plt.subplots(figsize = [20,10])


    order = ['bootHCP','sub300HCP','sub200HCP','HCP2K','HCP5K','HCP10K','HCPLOO','HCPTrain','bootPNC','sub300PNC','sub200PNC','PNC2K','PNC5K','PNC10K','PNCLOO','PNCTrain']


    #d = sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='modelTestSample',order = order,ax=ax,color='black',alpha = 0.4,inner=None,)

    #for vio in d.collections:
    #    vio.set_facecolor('black')
    #    vio.set_alpha(0.4)

    d = sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='modelTestSample',order = order,ax=ax,color='black',boxprops=dict(alpha=.5))

    d=sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSize],y="R Squared",x='modelTestSample',hue='sampleNum',ax=ax,order=order, boxprops=dict(alpha=.7))






    sns.despine(offset=15)

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='center right',bbox_to_anchor=(0.9, 0.55),fontsize=25,title = 'HCP Split',title_fontsize=25)
    ax.get_legend().remove()
    ax.set_ylim([-0.05,0.4])
    ax.set_xlim([-1,16])

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    #plt.title('Within Sample \t\t\t\t\t\t\t\t\t\t\t Out of Sample \t\t'.expandtabs(), fontsize=40)
    #plt.xticks(ticks = range(0,16), labels = ['Bagged', 'Subsample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only','Bagged', 'Subample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only'],rotation=25,fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Performance (R Squared)', fontsize=30)
    plt.xlabel('')
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.tight_layout()
    #plt.show()



    plt.savefig(os.path.join(globalOpdir,'92PerformanceComparisonSplit'+testSize+'.png'))
    plt.close()






allXhistBoot=np.stack(histDenGatherBoot)
allXhistMeanBoot=np.mean(allXhistBoot,axis=0)
allXhistStdBoot=np.std(allXhistBoot,axis=0)

allXhistS300=np.stack(histDenGatherS300)
allXhistMeanS300=np.mean(allXhistS300,axis=0)
allXhistStdS300=np.std(allXhistS300,axis=0)

allXhistS200=np.stack(histDenGatherS200)
allXhistMeanS200=np.mean(allXhistS200,axis=0)
allXhistStdS200=np.std(allXhistS200,axis=0)


#################################################
############ BS model thresholding ##############
#################################################


allThreshResDf =  pd.concat(threshPerformGather)

allThreshResDf['R Squared']=allThreshResDf['pearsonsR']**2


allThreshResDf = allThreshResDf[~((allThreshResDf.testSample == 'pnc') & (allThreshResDf.testSize == '400'))]

allThreshResDf.testSize.replace({'400':'All','787':'All'},inplace=True)


allThreshResDfPivot = allThreshResDf.pivot(index=['sampleNum','iter'],columns=['modelTestSample','testSize','thresh'],values='R Squared')



for testSampleSize in ['200','300','All']:
    for testSample in ['HCP','PNC']:

        f = plt.figure(figsize=[12,12])
        sns.set_style('white')

        baseXPositions = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]


        #################### Top Subplot ################################
        ax = f.add_subplot(311)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['boot'+testSample][testSampleSize].values.T],positions=baseXPositions,labels=labels1,widths=0.06,medianprops=medianprops)
        ax.set_ylim(-0.05,0.4)
        ax.set_ylabel('Performance (R Squared)')
        #ax.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax2 = ax.twinx()


        ax2.bar(np.arange(0.05,1.05,0.1), allXhistMeanBoot, yerr=allXhistStdBoot,alpha=0.3,width=0.1) 

        for i,rect in enumerate(allXhistMeanBoot):
            txt="{0:.1%}".format(rect)
            ax2.text((i/10)+0.05,rect+allXhistStdBoot[i],txt, ha='center', va='bottom',alpha=0.5)

        ax2.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax2.yaxis.set_major_formatter(formatter)



        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax.set_xlim(-0.05,1.05)

        ax.set_xlabel('Percentage of boostraps features occured in')
        ax.set_title('Bagged models performance with feature thresholding within sample ('+testSample+')')


        #################### Middle Subplot ################################
        ### S300 model threshold model in HCP
        #f = plt.figure(figsize=[10,6])
        ax3 = f.add_subplot(312)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub300'+testSample][testSampleSize].values.T],positions=baseXPositions,labels=labels2,widths=0.06,medianprops=medianprops)
        ax3.set_ylim(-0.05,0.4)
        ax3.set_ylabel('Performance (R Squared)')

        ax4 = ax3.twinx()


        x=ax4.bar(np.arange(0.05,1.05,0.1), allXhistMeanS300, yerr=allXhistStdS300,alpha=0.3,width=0.1) 

        for i,rect in enumerate(allXhistMeanS300):
            txt="{0:.1%}".format(rect)
            ax4.text((i/10)+0.05,rect+allXhistStdS300[i],txt, ha='center', va='bottom',alpha=0.5)

        ax4.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax4.yaxis.set_major_formatter(formatter)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax3.xaxis.grid(False)
        ax3.yaxis.grid(True)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax3.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax3.set_xlabel('Percentage of subsamples features occured in')
        ax3.set_title('Subsample 300 models performance with feature thresholding within sample ('+testSample+')')

        #################### Bottom Subplot ################################
        ### S200 model threshold model in HCP
        #f = plt.figure(figsize=[10,6])
        ax3 = f.add_subplot(313)
        medianprops = dict(linestyle='-', linewidth=1, color='black')
        labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
        ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub200'+testSample][testSampleSize].values.T],positions=baseXPositions,labels=labels2,widths=0.06,medianprops=medianprops)
        ax3.set_ylim(-0.05,0.4)
        ax3.set_ylabel('Performance (R Squared)')

        ax4 = ax3.twinx()


        x=ax4.bar(np.arange(0.05,1.05,0.1), allXhistMeanS200, yerr=allXhistStdS200,alpha=0.3,width=0.1) 

        for i,rect in enumerate(allXhistMeanS200):
            txt="{0:.1%}".format(rect)
            ax4.text((i/10)+0.05,rect+allXhistStdS200[i],txt, ha='center', va='bottom',alpha=0.5)

        ax4.set_ylabel('Percentage of total features included')

        # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)

        # Set the formatter
        ax4.yaxis.set_major_formatter(formatter)

        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax3.xaxis.grid(False)
        ax3.yaxis.grid(True)

        sns.despine(trim=True,left=False, bottom=False, right=False)
        ax3.set_xlim(-0.05,1.05)
        #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
        ax3.set_xlabel('Percentage of subsamples features occured in')
        ax3.set_title('Subsample 200 models performance with feature thresholding within sample ('+testSample+')')

        plt.tight_layout()
        plt.savefig(os.path.join(globalOpdir,'6'+testSample+'ThreshPerform'+testSampleSize+'.png'))
        plt.close()







###### Rval versus inclusion

bootRvalsAll = np.concatenate([rvalDict[r]['bootRvals'] for r in rvalDict])
bootEdgesAll = np.concatenate([rvalDict[r]['bootEdgeCount'] for r in rvalDict])
sub300RvalsAll = np.concatenate([rvalDict[r]['sub300Rvals'] for r in rvalDict])
sub200RvalsAll = np.concatenate([rvalDict[r]['sub200Rvals'] for r in rvalDict])
sub300EdgesAll = np.concatenate([rvalDict[r]['sub300EdgeCount'] for r in rvalDict])
sub200EdgesAll = np.concatenate([rvalDict[r]['sub200EdgeCount'] for r in rvalDict])


mlInd = pd.MultiIndex.from_tuples(zip(*[['boot','sub300','sub200','boot','sub300','sub200'],['edgeCount','edgeCount','edgeCount','Rvals','Rvals','Rvals']]))
arr=np.stack([bootEdgesAll,sub300EdgesAll,sub200EdgesAll,bootRvalsAll,sub300RvalsAll,sub200RvalsAll])
tempDf=pd.DataFrame(arr.T,columns=mlInd)

tempDfStack = tempDf.stack(level=0).reset_index()

tempDfStack=tempDfStack.rename({'level_1':'ModelType'},axis=1)

replaceDict={'boot':'Bagged','sub200':'Subsample 200','sub300':'Subsample 300'}
tempDfStack.ModelType.replace(replaceDict,inplace=True)
tempDfStack = tempDfStack[~(tempDfStack.edgeCount == 0)]
tempDfStack.rename({'Rvals':'R Value at feature selection step','edgeCount':'Feature ocurrence across resamples/bootstraps'},axis=1,inplace=True)


plt.figure(figsize=[12,18])
sns.set(style="white", palette="bright", color_codes=True)
s=sns.jointplot(data=tempDfStack,x='R Value at feature selection step', y='Feature ocurrence across resamples/bootstraps',hue='ModelType',alpha=0.05,linewidth=0,marker='.')
s.fig.gca().set_ylabel('Feature ocurrence across resamples/bootstraps')
s.fig.gca().set_xlabel('R Value at feature selection step')

#s.yaxis.grid(True)
#plt.grid(axis='y')
plt.ylim([0,1])
s.ax_joint.yaxis.grid(True)
sns.despine()

plt.tight_layout()

plt.savefig(os.path.join(globalOpdir,'5RvalVOccurenceAll.png'))

plt.close('all')





################## Thresh plots all in one fig



for testSampleSize in ['200','300','All']:
    baseXPosition = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]

    f = plt.figure(figsize=[12,12])
    sns.set_style('white')
    ax = f.add_subplot(311)
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
    labels1Blank = ['','','','','','','','','','']
    bp1 = ax.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['bootHCP'][testSampleSize].values.T],positions=baseXPosition,labels=labels1,widths=0.025,medianprops=medianprops,patch_artist=True,boxprops=dict(facecolor='white'))
    bp2 = ax.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['bootPNC'][testSampleSize].values.T],positions=[0.08,0.18,0.28,0.38,0.48,0.58,0.68,0.78,0.88,0.98],labels=labels1Blank,widths=0.025,medianprops=medianprops,patch_artist=True,boxprops=dict(facecolor='grey'))

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Within Sample (HCP)', 'Out of Sample (PNC)'], loc='upper right')




    ax.set_ylim(-0.05,0.4)
    ax.set_ylabel('Performance (R Squared)')
    #ax.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
    ax2 = ax.twinx()


    ax2.bar(np.arange(0.065,1.065,0.1), allXhistMeanBoot, yerr=allXhistStdBoot,alpha=0.3,width=0.1) 

    for i,rect in enumerate(allXhistMeanBoot):
        txt="{0:.1%}".format(rect)
        ax2.text((i/10)+0.065,rect+allXhistStdBoot[i],txt, ha='center', va='bottom',alpha=0.5)

    ax2.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax2.yaxis.set_major_formatter(formatter)



    sns.despine(trim=True,left=False, bottom=False, right=False)
    ax.set_xlim(-0.05,1.1)

    ax.set_xlabel('Percentage of boostraps features occured in')
    ax.set_title('Bagged models performance with feature thresholding')

    ### S300 model threshold model in HCP
    #f = plt.figure(figsize=[10,6])
    ax3 = f.add_subplot(312)
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
    ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub300HCP'][testSampleSize].values.T],positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.025,medianprops=medianprops)
    ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub300PNC'][testSampleSize].values.T],positions=[0.08,0.18,0.28,0.38,0.48,0.58,0.68,0.78,0.88,0.98],labels=labels1Blank,widths=0.025,medianprops=medianprops,patch_artist=True,boxprops=dict(facecolor='grey'))

    ax3.set_ylim(-0.05,0.4)
    ax3.set_ylabel('Performance (R Squared)')

    ax4 = ax3.twinx()


    x=ax4.bar(np.arange(0.065,1.065,0.1), allXhistMeanS300, yerr=allXhistStdS300,alpha=0.3,width=0.1) 

    for i,rect in enumerate(allXhistMeanS300):
        txt="{0:.1%}".format(rect)
        ax4.text((i/10)+0.065,rect+allXhistStdS300[i],txt, ha='center', va='bottom',alpha=0.5)

    ax4.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax4.yaxis.set_major_formatter(formatter)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax3.xaxis.grid(False)
    ax3.yaxis.grid(True)

    sns.despine(trim=True,left=False, bottom=False, right=False)
    ax3.set_xlim(-0.05,1.1)
    #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
    ax3.set_xlabel('Percentage of subsamples features occured in')
    ax3.set_title('Subsample 300 models performance with feature thresholding')


    ### S200 model threshold model in HCP
    #f = plt.figure(figsize=[10,6])
    ax3 = f.add_subplot(313)
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
    ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub200HCP'][testSampleSize].values.T],positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.025,medianprops=medianprops)
    ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub200PNC'][testSampleSize].values.T],positions=[0.08,0.18,0.28,0.38,0.48,0.58,0.68,0.78,0.88,0.98],labels=labels1Blank,widths=0.025,medianprops=medianprops,patch_artist=True,boxprops=dict(facecolor='grey'))

    ax3.set_ylim(-0.05,0.4)
    ax3.set_ylabel('Performance (R Squared)')

    ax4 = ax3.twinx()


    x=ax4.bar(np.arange(0.065,1.065,0.1), allXhistMeanS200, yerr=allXhistStdS200,alpha=0.3,width=0.1) 

    for i,rect in enumerate(allXhistMeanS200):
        txt="{0:.1%}".format(rect)
        ax4.text((i/10)+0.065,rect+allXhistStdS200[i],txt, ha='center', va='bottom',alpha=0.5)

    ax4.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax4.yaxis.set_major_formatter(formatter)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax3.xaxis.grid(False)
    ax3.yaxis.grid(True)

    sns.despine(trim=True,left=False, bottom=False, right=False)
    ax3.set_xlim(-0.05,1.1)
    #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
    ax3.set_xlabel('Percentage of subsamples features occured in')
    ax3.set_title('Subsample 200 models performance with feature thresholding')

    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'6HCPandPNCThreshPerform'+testSampleSize+'.png'))
    plt.close()




#########################################################################
######################## Strip plots ####################################
#########################################################################


################################ Figures of All


for testSampleSize in ['200','300','All']:

    if testSampleSize == 'All':
        swarmSize = 2
    else:
        swarmSize = 0.5

    ### Bootstrap model performance all edges, Single model vs BS model, test sample size 200
    fig, ax = plt.subplots(figsize=[8,6])


    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    #d = sns.swarmplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='testSample',hue='modelType',order = ['HCP','PNC'],dodge=True,size=swarmSize,ax=ax,alpha = 1,color='black')



    d = sns.violinplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='testSample',hue='modelType',order = ['HCP','PNC'],alpha = 1,inner=None,linewidth=0,ax=ax)

    #d = sns.stripplot(data=iterResPlotDf2[iterResPlotDf2.testSize == '200'],y="R Squared",x='testSample',hue='modelType',order = ['HCP','PNC'],ax=ax,dodge=True,jitter = 1)

    for vio in d.collections:
        vio.set_facecolor('black')
        vio.set_alpha(0.25)


    d = sns.boxplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSampleSize],y="R Squared",x='testSample',hue='modelType',order = ['HCP','PNC'],ax=ax, boxprops=dict(alpha=.9),showfliers=False)





    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[8:], labels[8:])

    cols=np.zeros([1,20])+1
    cols[:,10:] = 2

    d.pcolorfast((-1,2), (-0.3,0.7),cols,cmap='brg', alpha=0.1)


    fracOff=1/3.72

    #plt.plot([-fracOff, -fracOff, 0, 0], [0.6, 0.62, 0.62, 0.6], lw=1.5, c='k')
    #plt.text(-fracOff/2, 0.62, "*", ha='center', va='bottom', color='k')
    #plt.plot([fracOff, fracOff, 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text(0, 0.61, "*", ha='center', va='bottom', color='k',weight="bold")

    #plt.plot([1-fracOff, 1-fracOff, 1+fracOff, 1+fracOff], [0.45, 0.47, 0.47, 0.45], lw=1.5, c='k')
    #plt.text(1+fracOff, 0.42, "*", ha='center', va='bottom', color='k',weight="bold")
    #plt.plot([(1/3.8), (1/3.8), 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text((1/7.6), 0.59, "*", ha='center', va='bottom', color='k')


    sns.despine(left=True, bottom=True)
    #plt.set_title('Performance of all models within and out of sample', fontsize=14)
    #ax.set_title('Within Sample \t\t\t\t\t\t\t\t Out of Sample \t\t'.expandtabs(), fontsize=40)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.ylim([-0.05,0.4])
    plt.xlim([-0.5,1.5])
    plt.ylabel('Performance (R Squared)')
    plt.xticks([0,1],['Within Sample (HCP)','Out of Sample (PNC)'])
    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'93BootstrapComparisonStripplotAllTestSample'+testSampleSize+'.png'))
    plt.close()






fig, ax = plt.subplots(figsize=[20,10])


sns.set(style="whitegrid", palette="pastel", color_codes=True)


d = sns.catplot(x="modelType", y="R Squared",hue="testSize", col="testSample",data=iterResPlotDf2, kind="strip",col_order = ['HCP','PNC'],legend = False)

d.set_xticklabels(rotation=30)


sns.despine(left=True, bottom=True)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
plt.ylim([-0.05,0.4])
#plt.xlim([-0.5,1.5])
#plt.ylabel('Performance (R Squared)')
plt.legend(loc='upper right',title = 'Test Sample Size')
plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'93BootstrapComparisonStripplotAllTestSample3TestSize.png'))
plt.close()





###### Performance by split grid form



#f = plt.figure()

d = sns.catplot(x="modelType", y="R Squared",hue="sampleNum", col="testSample",row = 'testSize',data=iterResPlotDf2, kind="strip",col_order = ['HCP','PNC'],legend = False,height=5,aspect=2,sharex=False)
#sns.despine(offset=15)
d.set_xticklabels(rotation=15)



plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'92PerformanceComparisonSplitGridStrip.png'))
plt.close()



###### Figure performance by split, each test size separate
for testSize in ['200','300','All']:
    f = plt.figure(figsize=[40,18])
    gs  = matplotlib.gridspec.GridSpec(1, 1, right=0.85)

    #ax = f.add_subplot(211)
    ax=plt.subplot(gs[0])
    order = ['bootHCP','sub300HCP','sub200HCP','HCP2K','HCP5K','HCP10K','HCPLOO','HCPTrain','bootPNC','sub300PNC','sub200PNC','PNC2K','PNC5K','PNC10K','PNCLOO','PNCTrain']
    d=sns.stripplot(data=iterResPlotDf2[iterResPlotDf2.testSize == testSize],y="R Squared",x='modelTestSample',hue='sampleNum',ax=ax,order=order)
    cols=np.zeros([1,160])+1
    cols[:,80:] = 2


    d.pcolorfast((-1,16), (-0.05,0.4),cols,cmap='brg', alpha=0.1)
    sns.despine(offset=15)
    #ax3.legend(bbox_to_anchor=(1.8, 1.5), loc='upper right')
    #ax3.legend(bbox_to_anchor=(1.3, 1.5), loc='upper right')
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='center right',bbox_to_anchor=(0.9, 0.55),fontsize=25,title = 'HCP Split',title_fontsize=25)
    ax.get_legend().remove()
    ax.set_ylim([-0.05,0.4])
    ax.set_xlim([-1,16])

    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.title('Within Sample \t\t\t\t\t\t\t\t\t\t\t Out of Sample \t\t'.expandtabs(), fontsize=40)
    plt.xticks(ticks = range(0,16), labels = ['Bagged', 'Subsample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only','Bagged', 'Subample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only'],rotation=25,fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Performance (R Squared)', fontsize=30)
    plt.xlabel('')
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'92PerformanceComparisonSplit'+testSize+'Strip.png'))
    plt.close()



labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']

allXhistBoot=np.stack(histDenGatherBoot)
bootHistDf = pd.DataFrame(allXhistBoot,columns = labels1)

allXhistMeanBoot=np.mean(allXhistBoot,axis=0)
allXhistStdBoot=np.std(allXhistBoot,axis=0)

allXhistS300=np.stack(histDenGatherS300)
s300HistDf = pd.DataFrame(allXhistS300,columns = labels1)

allXhistMeanS300=np.mean(allXhistS300,axis=0)
allXhistStdS300=np.std(allXhistS300,axis=0)

allXhistS200=np.stack(histDenGatherS200)
s200HistDf = pd.DataFrame(allXhistS200,columns = labels1)

allXhistMeanS200=np.mean(allXhistS200,axis=0)
allXhistStdS200=np.std(allXhistS200,axis=0)


#################################################
############ BS model thresholding ##############
#################################################


allThreshResDf =  pd.concat(threshPerformGather)

allThreshResDf['R Squared']=allThreshResDf['pearsonsR']**2


allThreshResDf = allThreshResDf[~((allThreshResDf.testSample == 'pnc') & (allThreshResDf.testSize == '400'))]

allThreshResDf.testSize.replace({'400':'All','787':'All'},inplace=True)


allThreshResDfPivot = allThreshResDf.pivot(index=['sampleNum','iter'],columns=['modelTestSample','testSize','thresh'],values='R Squared')








################## Thresh plots all in one fig

plt.close('all')
plt.clf()

for testSampleSize in ['200','300','All']:

    if testSampleSize == 'All':
        swarmSize = 3
    else:
        swarmSize = 0.8

    baseXPosition = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]

    f = plt.figure(figsize=[16,16])

    ax = f.add_subplot(311)
    sns.set(font_scale=1.4)
    sns.set_style('white')

    sns.swarmplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'boot')],y='R Squared',x='thresh',hue='testSample',dodge=True,size=swarmSize,ax=ax,alpha = 0.6,color='black',hue_order=['hcp','pnc'])

    d = sns.boxplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'boot')],y='R Squared',x='thresh',hue='testSample',ax=ax, boxprops=dict(alpha=.95),hue_order=['hcp','pnc'],palette = sns.color_palette(palette = ["white" , "grey"]),showfliers=False)

    #for i,bx in enumerate(d.artists):
    #    if i % 2:
    #        bx.set_facecolor('grey')
    #    else:
    #        bx.set_facecolor('white')
 



    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Within Sample (HCP)', 'Out of Sample (PNC)'], loc='upper right')


    ax.set_ylim(-0.05,0.4)
    ax.set_ylabel('Performance (R Squared)')
    ax2 = ax.twinx()

    sns.barplot(data=bootHistDf,alpha = 0.3,color = 'blue',ax=ax2)

    for i,rect in enumerate(allXhistMeanBoot):
        txt="{0:.1%}".format(rect)
        ax2.text((i)+0.065,rect+allXhistStdBoot[i],txt, ha='center', va='bottom',alpha=0.5)

    ax2.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax2.yaxis.set_major_formatter(formatter)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)


    sns.despine(trim=True,left=False, bottom=False, right=False)
    #ax.set_xlim(-0.05,1.1)

    ax.set_xlabel('Percentage of boostraps features occured in')
    ax.set_title('Bagged models performance with feature thresholding')

    ### S300 model threshold model in HCP
    #f = plt.figure(figsize=[10,6])
    ax3 = f.add_subplot(312)

    sns.set_style('white')



    sns.swarmplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'sub300')],y='R Squared',x='thresh',hue='testSample',dodge=True,size=swarmSize,ax=ax3,alpha = 0.6,color='black',hue_order=['hcp','pnc'])

    sns.boxplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'sub300')],y='R Squared',x='thresh',hue='testSample',ax=ax3, boxprops=dict(alpha=.95),hue_order=['hcp','pnc'],palette = sns.color_palette(palette = ["white" , "grey"]),showfliers=False)

    handles, labels = ax3.get_legend_handles_labels()
    #ax3.legend(handles[:2], labels[:2])
    ax3.get_legend().remove()


    ax3.set_ylim(-0.05,0.4)
    ax3.set_ylabel('Performance (R Squared)')

    ax4 = ax3.twinx()


    sns.barplot(data=s300HistDf,alpha = 0.3,color = 'blue',ax=ax4)

    for i,rect in enumerate(allXhistMeanS300):
        txt="{0:.1%}".format(rect)
        ax4.text((i)+0.065,rect+allXhistStdS300[i],txt, ha='center', va='bottom',alpha=0.5)

    ax4.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax4.yaxis.set_major_formatter(formatter)

    ax3.xaxis.grid(False)
    ax3.yaxis.grid(True)
    ax4.xaxis.grid(False)
    ax4.yaxis.grid(False)

    sns.despine(trim=True,left=False, bottom=False, right=False)
    #ax3.set_xlim(-0.05,1.1)
    #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
    ax3.set_xlabel('Percentage of subsamples features occured in')
    ax3.set_title('Subsample 300 models performance with feature thresholding')


    ### S200 model threshold model in HCP
    #f = plt.figure(figsize=[10,6])
    ax3 = f.add_subplot(313)

    sns.set_style('white')
    sns.swarmplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'sub200')],y='R Squared',x='thresh',hue='testSample',dodge=True,size=swarmSize,ax=ax3,alpha = 0.6,color='black',hue_order=['hcp','pnc'])

    sns.boxplot(data = allThreshResDf[(allThreshResDf.testSize == testSampleSize) & (allThreshResDf.modelType == 'sub200')],y='R Squared',x='thresh',hue='testSample',ax=ax3, boxprops=dict(alpha=.95),hue_order=['hcp','pnc'],palette = sns.color_palette(palette = ["white" , "grey"]),showfliers=False)

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[:2], labels[:2])
    ax3.get_legend().remove()

    ax3.set_ylim(-0.05,0.4)
    ax3.set_ylabel('Performance (R Squared)')

    ax4 = ax3.twinx()


    sns.barplot(data=s200HistDf,alpha = 0.3,color = 'blue',ax=ax4)

    for i,rect in enumerate(allXhistMeanS200):
        txt="{0:.1%}".format(rect)
        ax4.text((i)+0.065,rect+allXhistStdS200[i],txt, ha='center', va='bottom',alpha=0.5)

    ax4.set_ylabel('Percentage of total features included')

    # https://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    ax4.yaxis.set_major_formatter(formatter)


    ax3.xaxis.grid(False)
    ax3.yaxis.grid(True)
    ax4.xaxis.grid(False)
    ax4.yaxis.grid(False)

    sns.despine(trim=True,left=False, bottom=False, right=False)
    #ax3.set_xlim(-0.05,1.1)
    #ax3.set_xticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%'])
    ax3.set_xlabel('Percentage of subsamples features occured in')
    ax3.set_title('Subsample 200 models performance with feature thresholding')

    plt.tight_layout()
    plt.savefig(os.path.join(globalOpdir,'96HCPandPNCThreshPerform'+testSampleSize+'Swarm.png'))
    plt.close()








