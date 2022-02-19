

import math
import torch as th
from einops import rearrange,repeat

import vpss

def optional(pydict,key,default):
    if pydict is None: return default
    if not(key in pydict): return default
    else: return pydict[key]

def needle_subset(data,sigma,snum,**kwargs):


    # -- optional --
    pt = optional(kwargs,'pt',1)
    c = optional(kwargs,'c',3)
    ps = optional(kwargs,'ps',23)

    # -- expand patches --
    print("ps: ",ps)
    n,k,pdim = data.shape
    edata = rearrange(data,'n k (t c h w) -> n k t c h w',t=pt,c=c,h=ps,w=ps)

    # -- build needles --
    nps = 3
    nscales = 8
    scale = 0.75
    needles = vpss.get_needles(edata,nps,nscales,scale)
    # print("[1] needles.shape: ",needles.shape)
    # needles = needles[:,:,[0]]
    # print("[2] needles.shape: ",needles.shape)

    # -- compute values --
    mean_dims = (-5,-4,-3,-2,-1)
    vals = th.mean((needles[:,[0]] - needles)**2,mean_dims)

    # -- sort inds --
    inds = th.argsort(vals,1)

    # -- subset --
    vals = vals[:,:snum]
    inds = inds[:,:snum]

    return vals,inds
