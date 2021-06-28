import numpy as np
import torch
import pandas as pd
from ufuncs import calPearsonR,rank
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
