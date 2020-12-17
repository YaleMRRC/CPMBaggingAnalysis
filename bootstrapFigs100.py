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

    globalIpdir = 'path_to/iter'+str(itr).zfill(2)
    globalOpdir = 'path_to/figs/'

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



    iterResPath = os.path.join(globalIpdir,'iterRes.npy')
    iterResDict = np.load(iterResPath, allow_pickle = True).item()


    ## iterResDict Structure
    #resDict[perfIter]['bootHCP'] = bootRHCP
    #resDict[perfIter]['sub300HCP'] = sub300RHCP
    #resDict[perfIter]['sub200HCP'] = sub200RHCP
    #resDict[perfIter]['HCP2K'] = gatherHCP2K
    #resDict[perfIter]['HCP5K'] = gatherHCP5K
    #resDict[perfIter]['HCP10K'] = gatherHCP10K
    #resDict[perfIter]['HCPLOO'] = gatherHCPLOO
    #resDict[perfIter]['HCPTrain'] = trainRHCP

    #resDict[perfIter]['bootPNC'] = bootRPNC
    #resDict[perfIter]['sub300PNC'] = sub300RPNC
    #resDict[perfIter]['sub200PNC'] = sub200RPNC
    #resDict[perfIter]['PNC2K'] = gatherPNC2K
    #resDict[perfIter]['PNC5K'] = gatherPNC5K
    #resDict[perfIter]['PNC10K'] = gatherPNC10K
    #resDict[perfIter]['PNCLOO'] = gatherPNCLOO
    #resDict[perfIter]['PNCTrain'] = trainRPNC



    resDictArray = [[iterResDict[k1][k2] for k2 in iterResDict[k1].keys()] for k1 in iterResDict.keys()]

    resDictList = [np.stack([x1[i] for x1 in resDictArray],axis=0).flatten() for i in range(0,16)]





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

    arrays=[['Bagged', 'sub300Resample', 'sub200Resample', 'SplitHalf', 'FiveFold', 'TenFold', 'LOO', 'TrainOnly']*2,['HCP->HCP','HCP->HCP','HCP->HCP','HCP->HCP','HCP->HCP','HCP->HCP','HCP->HCP','HCP->HCP','HCP->PNC','HCP->PNC','HCP->PNC','HCP->PNC','HCP->PNC','HCP->PNC','HCP->PNC','HCP->PNC']]


    ### Bootstrap model performance all edges, Single model vs BS model


    #arrays=[['S200','S400','Bagged','S200','S400','Bagged'],['Within Sample','Within Sample','Within Sample','Out of Sample','Out of Sample','Out of Sample',]]
    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=['ModelType', 'TestSample'])
    df=pd.DataFrame(resDictList,index=index).T

    dfUnstack=df.unstack().reset_index()
    dfUnstack=dfUnstack.rename({0:'Pearsons R'},axis=1)   
    dfUnstack.dropna(inplace=True)
    dfUnstack.reset_index(inplace=True)

    dfUnstack['combo'] = dfUnstack.ModelType.values+' '+dfUnstack['TestSample'].values

    dfLen = dfUnstack.shape[0]
    dfUnstack['sampleNum'] = np.repeat([itr],dfLen)



    iterResGather.append(dfUnstack)


    fig, ax = plt.subplots(figsize=[8,6])
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    d=sns.violinplot(data=dfUnstack,y="Pearsons R",x='TestSample',inner='quartile',hue='ModelType') 


    #fracOff=1/3.72

    #plt.plot([-fracOff, -fracOff, 0, 0], [0.6, 0.62, 0.62, 0.6], lw=1.5, c='k')
    #plt.text(-fracOff/2, 0.62, "*", ha='center', va='bottom', color='k')
    #plt.plot([fracOff, fracOff, 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text(0, 0.56, "*", ha='center', va='bottom', color='k',weight="bold")

    #plt.plot([1-fracOff, 1-fracOff, 1+fracOff, 1+fracOff], [0.45, 0.47, 0.47, 0.45], lw=1.5, c='k')
    #plt.text(1+fracOff, 0.43, "*", ha='center', va='bottom', color='k',weight="bold")
    #plt.plot([(1/3.8), (1/3.8), 0, 0], [0.57, 0.59, 0.59, 0.57], lw=1.5, c='k')
    #plt.text((1/7.6), 0.59, "*", ha='center', va='bottom', color='k')


    sns.despine(left=True, bottom=False)
    ax.set_title('Performance of single and bootstrapped models within and out of sample', fontsize=14)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.ylim([-0.2,0.65])
    plt.xlim([-0.4,1.4])
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

    threshResOppath = os.path.join(globalIpdir,'threshRes.npy')
    threshResDict = np.load(threshResOppath, allow_pickle = True).item()

    ## threshResDict Structure

    #resDict['bootHCP'][thresh].append(bootRHCP)
    #resDict['sub300HCP'][thresh].append(sub300RHCP)
    #resDict['sub200HCP'][thresh].append(sub200RHCP)

    #resDict['bootPNC'][thresh].append(bootRPNC)
    #resDict['sub300PNC'][thresh].append(sub300RPNC)
    #resDict['sub200PNC'][thresh].append(sub200RPNC)

    #### Creating DataFrame
    threshsDf = list(map(lambda x : round (x,2), np.arange(0,1,0.1)))
    lowerIndDf = ['>='+str(t) if t > 0 else '>'+str(t) for t in threshsDf]

    arrays=[np.repeat(np.array(list(threshResDict.keys())),10),np.tile(np.array(lowerIndDf),6)]


    ### Bootstrap model performance all edges, Single model vs BS model

    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=['ModelType', 'Threshold'])

    threshResDictArr = np.stack([threshResDict[k1][k2] for k1 in threshResDict.keys() for k2 in threshResDict[k1].keys()])

    threshDf=pd.DataFrame(threshResDictArr,index=index).T

    dfUnstackThresh=threshDf.unstack().reset_index()

    dfUnstackThresh=dfUnstackThresh.rename({0:'Pearsons R'},axis=1)   

    dfUnstackThresh.reset_index(inplace=True)

    dfLen = dfUnstackThresh.shape[0]
    dfUnstackThresh['sampleNum'] = np.repeat([itr],dfLen)


    threshPerformGather.append(dfUnstackThresh)






    f = plt.figure(figsize=[10,12])
    sns.set_style('white')
    ax = f.add_subplot(311)
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
    ax.boxplot(np.stack(threshResDict['bootHCP'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
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
    ax3.boxplot(np.stack(threshResDict['sub300HCP'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
    ax5.boxplot(np.stack(threshResDict['sub200HCP'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
    plt.savefig(os.path.join(globalOpdir,'HCPThreshPerform'+str(itr)+'.png'))
    plt.close()



    f = plt.figure(figsize=[10,12])
    sns.set_style('white')
    ax = f.add_subplot(311)
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
    ax.boxplot(np.stack(threshResDict['bootPNC'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
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
    ax3.boxplot(np.stack(threshResDict['sub300PNC'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
    ax5.boxplot(np.stack(threshResDict['sub200PNC'].values()).T,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
    plt.savefig(os.path.join(globalOpdir,'PNCThreshPerform'+str(itr)+'.png'))
    plt.close()



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


    rvalDict[itr]['bootRvals'] = np.mean(bootRes[5],axis=0)
    rvalDict[itr]['sub300Rvals'] = np.mean(sub300Res[5],axis=0)
    rvalDict[itr]['sub200Rvals'] = np.mean(sub200Res[5],axis=0)

    rvalDict[itr]['bootEdgeCount'] = np.stack(bootRes[1]).mean(axis=0)+np.stack(bootRes[2]).mean(axis=0)
    rvalDict[itr]['sub300EdgeCount'] = np.stack(sub300Res[1]).mean(axis=0)+np.stack(sub300Res[2]).mean(axis=0)
    rvalDict[itr]['sub200EdgeCount'] = np.stack(sub200Res[1]).mean(axis=0)+np.stack(sub200Res[2]).mean(axis=0)


allIterResDf =  pd.concat(iterResGather)


allIterResDf['R Squared']=allIterResDf['Pearsons R']**2


# Mean Perf
# allIterResDf.drop(['index','level_2','Pearsons R','combo','sampleNum'],axis=1).groupby(['ModelType','TestSample']).mean()
#  reshapedDf = allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')
#allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')
#allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')['LOO'].mean()

allThreshResDf =  pd.concat(threshPerformGather)



################################ Summary Tables

allIterResDf.drop(['index','level_2','Pearsons R','combo','sampleNum'],axis=1).groupby(['ModelType','TestSample']).mean()
reshapePerfDf = allIterResDf.drop(['index','level_2','Pearsons R','combo'],axis=1).reset_index().pivot(columns = ['ModelType','TestSample'],values='R Squared')
meanPerfTable = reshapePerfDf.mean().sort_values(ascending=False)









################################ Figures of All

f = plt.figure(figsize=[24,10])

gs  = matplotlib.gridspec.GridSpec(1, 1, right=0.77)
ax=plt.subplot(gs[0])



### Bootstrap model performance all edges, Single model vs BS model
fig, ax = plt.subplots(figsize=[8,6])


sns.set(style="whitegrid", palette="pastel", color_codes=True)
allIterResDf.replace({'sub300Resample':'Subsample 300','sub200Resample': 'Subsample 200'},inplace=True)
d = sns.boxplot(data=allIterResDf,y="R Squared",x='TestSample',hue='ModelType')



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
plt.savefig(os.path.join(globalOpdir,'BootstrapComparisonBoxplotAll.png'))
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
sns.boxplot(data = allIterResDf[allIterResDf.TestSample == 'HCP->HCP'], x = 'ModelType', y = 'R Squared',color='white',order = ['SplitHalf','FiveFold', 'TenFold', 'LOO', 'TrainOnly','Subsample 200','Subsample 300','Bagged'])
#plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bootstrap','Train Only'])
plt.ylim([-0.1,0.3])
plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

plt.subplot(3,1,3)
plt.title('Performance on PNC')
sns.boxplot(data = allIterResDf[allIterResDf.TestSample == 'HCP->PNC'], x = 'ModelType', y = 'R Squared',color='white',order = ['SplitHalf','FiveFold', 'TenFold', 'LOO', 'TrainOnly','Subsample 200','Subsample 300','Bagged'])
#plt.xticks(range(1,9),['Split Half','Five Fold','Ten Fold','Leave One Out','subsample 200','subsample 300','Bagged','Train Only'])
plt.ylim([-0.1,0.3])
plt.grid(b=True,axis='y',alpha=0.7,linestyle='--')
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')

plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'allPerfIterSqViolin.png'))






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

os.system('montage '+opname1+' '+opname2+' '+opname3+' -geometry +3+1 '+os.path.join(globalOpdir,'histogramAllMontage.png'))



###### Figure

f = plt.figure(figsize=[40,18])
gs  = matplotlib.gridspec.GridSpec(1, 1, right=0.85)

#ax = f.add_subplot(211)
ax=plt.subplot(gs[0])
d=sns.boxplot(data=allIterResDf,y="R Squared",x='combo',hue='sampleNum',ax=ax)
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
plt.xticks(ticks = range(0,16), labels = ['Bagged', 'Resample 300', 'Resample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only','Bagged', 'Subample 300', 'Subsample 200', 'SplitHalf','FiveFold', 'TenFold', 'LOO', 'Train Only'],rotation=25,fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Performance (R Squared)', fontsize=30)
plt.xlabel('')
plt.gcf().subplots_adjust(bottom=0.25)

plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'PerformanceComparisonSplit.png'))
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

allThreshResDf.drop('index',axis=1,inplace=True)
allThreshResDf.drop('level_2',axis=1,inplace=True)
allThreshResDf.reset_index(inplace=True)
allThreshResDf['index'] = np.tile(range(1,101),1200)
allThreshResDf['R Squared']=allThreshResDf['Pearsons R']**2

allThreshResDfPivot = allThreshResDf.pivot(index=['index','sampleNum'],columns=['ModelType','Threshold'],values='R Squared')



f = plt.figure(figsize=[12,12])
sns.set_style('white')
ax = f.add_subplot(311)
medianprops = dict(linestyle='-', linewidth=1, color='black')
labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax.boxplot(allThreshResDfPivot['bootHCP'].values,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
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
ax.set_title('Bagged models performance with feature thresholding within sample (HCP)')

### S300 model threshold model in HCP
#f = plt.figure(figsize=[10,6])
ax3 = f.add_subplot(312)
medianprops = dict(linestyle='-', linewidth=1, color='black')
labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax3.boxplot(allThreshResDfPivot['sub300HCP'].values,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
ax3.set_title('Subsample 300 models performance with feature thresholding within sample (HCP)')


### S200 model threshold model in HCP
#f = plt.figure(figsize=[10,6])
ax3 = f.add_subplot(313)
medianprops = dict(linestyle='-', linewidth=1, color='black')
labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub200HCP'].values.T],positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
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
ax3.set_title('Subsample 200 models performance with feature thresholding within sample (HCP)')

plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'HCPThreshPerformAll.png'))
plt.close()










f = plt.figure(figsize=[12,12])
sns.set_style('white')
ax = f.add_subplot(311)
medianprops = dict(linestyle='-', linewidth=1, color='black')
labels1=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax.boxplot(allThreshResDfPivot['bootPNC'].values,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels1,widths=0.06,medianprops=medianprops)
ax.set_ylim(-0.05,0.25)
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
ax.set_title('Bootstrap models performance with feature thresholding out of sample (PNC)')

### S300 model threshold model in HCP
#f = plt.figure(figsize=[10,6])
ax3 = f.add_subplot(312)
medianprops = dict(linestyle='-', linewidth=1, color='black')
labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax3.boxplot(allThreshResDfPivot['sub300PNC'].values,positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)
ax3.set_ylim(-0.05,0.25)
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
ax3.set_xlabel('Percentage of boostraps features occured in')
ax3.set_title('Subsample 300 models performance with feature thresholding out of sample (PNC)')


### S200 model threshold model in HCP
#f = plt.figure(figsize=[10,6])
ax3 = f.add_subplot(313)
medianprops = dict(linestyle='-', linewidth=1, color='black')

labels2=['>0%','>=10%','>=20%','>=30%','>=40%','>=50%','>=60%','>=70%','>=80%','>=90%']
ax3.boxplot([a[~np.isnan(a)] for a in allThreshResDfPivot['sub200PNC'].values.T],positions=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],labels=labels2,widths=0.06,medianprops=medianprops)

#subdf = allThreshResDfPivot['sub200PNC']
#subdfStack = subdf.stack().reset_index()
#subdfStack.rename({0:'Pearsons R'},axis=1,inplace=True)

#sns.boxplot(ax=ax3, data = subdfStack,x='Threshold', y='Pearsons R')

ax3.set_ylim(-0.05,0.25)
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
ax3.set_xlabel('Percentage of boostraps features occured in')
ax3.set_title('Subsample 200 models performance with feature thresholding out of sample (PNC)')

plt.tight_layout()
plt.savefig(os.path.join(globalOpdir,'PNCThreshPerformAll25YLim.png'))
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

plt.savefig(os.path.join(globalOpdir,'RvalVOccurenceAll.png'))

plt.close('all')

