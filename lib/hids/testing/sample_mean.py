
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def sample_mean_cat(mean_cat,mean,mbounds,bsize,num,dim):
    if mean_cat == "fixed":
        return sample_mean_fixed(mean,bsize,num,dim)
    elif mean_cat == "lb-ub":
        return sample_mean_lbub(mean,mbounds,bsize,num,dim)
    else:
        raise ValueError(f"Uknown mean category [{cov_cat}]")

def sample_mean_fixed(mean,bsize,num,dim):
    # -- ensure tensor --
    if isinstance(mean,float) or isinstance(mean,int):
        mean = th.FloatTensor([mean])
        mean = repeat(mean,'1 -> d',d=dim)
    if not(th.is_tensor(mean)):
        mean = th.FloatTensor(mean)
        if mean.dim() == 1 and len(mean) == 1:
            mean = repeat(mean,'1 -> d',d=dim)
    if mean.dim() == 1:
        mean = mean[None,None,:]
    return mean

def sample_mean_lbub(mean,mbounds,bsize,num,dim):

    # -- sample unif [0,1] --
    lb = mbounds[0]
    ub = mbounds[1]
    samples = th.rand(bsize,num,dim)

    # -- unif [0,b-a] --
    samples = samples * (ub - lb)

    # -- unif [a,b] --
    samples = samples + lb

    return samples

