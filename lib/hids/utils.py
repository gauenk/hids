
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def compare_inds(gt,prop):
    delta = th.abs(gt[:,None,:] - prop[:,:,None])
    delta = th.any(delta < 1e-8,2)
    delta = delta.type(th.float)
    delta = delta.mean(1)
    delta = delta.mean(0)
    return delta

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

def gather_data(data,inds):
    """

    data.shape = (B,N,D)
    inds.shape = (B,N)

    """
    bsize,num,dim = data.shape
    inds = repeat(inds,'b n -> b n d',d=dim)
    rdata = th.gather(data,1,inds)
    return rdata
