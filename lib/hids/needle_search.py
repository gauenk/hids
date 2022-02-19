

import math
import torch as th
from einops import rearrange,repeat

import vpss

def needle_subset(data,sigma,snum,pt=1,c=3,ps=7):


    # -- expand patches --
    print("data.shape: ",data.shape)
    n,k,pdim = data.shape
    edata = rearrange(data,'n k (t c h w) -> n k t c h w',t=pt,c=c,h=ps,w=ps)
    print("edata.shape: ",edata.shape)

    # -- build needles --
    nps = 23
    nscales = 8
    scale = 0.75
    needles = vpss.get_needles(edata,nps,nscales,scale)

    # -- compute values --
    mean_dims = (-5,-4,-3,-2,-1)
    vals = th.mean((needles[:,[0]] - needles)**2,mean_dims)
    print("vals.shape: ",vals.shape)

    # -- sort inds --
    inds = th.argsort(vals,1)

    return vals,inds
