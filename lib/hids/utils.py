
# -- python --
import math

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def compute_target_sigma(sigma,m):
    var = sigma**2
    t_sigma2 = ((m-1)/m)**2 * var + (m-1)/(m**2) * var
    t_sigma = math.sqrt(t_sigma2)
    return t_sigma

def compare_inds(gt,prop,mbatch=True):
    delta = th.abs(gt[:,None,:] - prop[:,:,None])
    delta = th.any(delta < 1e-8,2)
    delta = delta.type(th.float)
    delta = delta.mean(1) # normalized by Number of Guesses
    if mbatch:
        delta = delta.mean(0)
        return delta.item()
    else:
        return delta

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

def gather_data(data,inds):
    """

    data.shape = (B,N,D)
    inds.shape = (B,N_s)
    N_s <= N

    """
    bsize,num,dim = data.shape
    inds = repeat(inds,'b n -> b n d',d=dim)
    rdata = th.gather(data,1,inds)
    return rdata

def clone(array):
    if th.is_tensor(array):
        return array.clone()
    elif isinstance(array,np.ndarray):
        return array.copy()
    else:
        tarray = type(array)
        raise TypeError(f"Uknown array type [{tarray}]")
