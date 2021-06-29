import numpy as np
import torch
import pandas as pd
from ufuncs import calPearsonR, rank
import seaborn as sns
from matplotlib import pyplot as plt


def from_ics_calc_ic_ir(ics, mode):
    if mode == "year":
        ic = np.nanmean(np.array(ics))
        icir = (np.nanmean(np.array(ics)) / np.nanstd(np.array(ics)))
        return ic, icir
    elif mode == "month":
        ics = np.array(ics)
        m = int(ics.shape[0] / 12)
        icirs = []
        for i in range(12):
            ic = np.nanmean(ics[i * m:(i + 1) * m])
            icir = (np.nanmean(ics[i * m:(i + 1) * m]) / np.nanstd(ics[i * m:(i + 1) * m]))
            icirs.append(np.array([ic, icir]))
        return np.stack(icirs)


def get_ic(pre,label,mode):
    corr=calPearsonR(pre.transpose(),label.transpose())
    if mode=="year":
        ic = np.nanmean(corr)
        icir = (np.nanmean(corr)/np.nanstd(corr))
        rankcorr=calPearsonR(rank(pre).transpose(),rank(label).transpose())
        rankic =  np.nanmean(rankcorr)
        rankicir = ( np.nanmean(rankcorr) / np.nanstd(rankcorr))
        return np.array([ic,icir,rankic,rankicir])
    elif mode=="month":
        ics=[]
        m=int(pre.shape[0]/12)
        for i in range(12):
            ppre = pre[i * m:(i + 1) * m, :]
            plabel = label[i * m:(i + 1) * m, :]

            corr = calPearsonR(ppre.transpose(), plabel.transpose())
            ic = np.nanmean(corr)
            icir = (np.nanmean(corr) / np.nanstd(corr))
            rankcorr = calPearsonR(rank(ppre).transpose(), rank(plabel).transpose())
            rankic = np.nanmean(rankcorr)
            rankicir = (np.nanmean(rankcorr) / np.nanstd(rankcorr))
            ics.append(np.array([ic,icir,rankic,rankicir]))
    return np.stack(ics)


def save_ics(icslist,index):
    icslist=pd.DataFrame(icslist,columns=['IC','ICIR','RankIC','RankICIR'],index=index)
    icslist.to_csv("ics.csv")
    return icslist
#sns.heatmap(table.loc[:,["IC","RankIC"]],annot=True, cmap="Reds")


def draw():
    table = pd.read_csv("records/IC_by_month.csv", index_col=0)
    table = pd.DataFrame(np.round(np.array(table), decimals=2), index=table.index, columns=table.columns)
    plt.figure(figsize=(18, 2))
    sns.heatmap(table.transpose(),
                annot=True,
                annot_kws={'size': 15, 'weight': 'bold'},
                cmap="Reds",
                cbar=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig('records/IC_heatmap.pdf')
    plt.show()
