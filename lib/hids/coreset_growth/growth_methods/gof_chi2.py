
# -- python --
import math

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- nn --
import torch.nn as nn

# -- local imports --
from .sample_weights import WeightNet

def gof_chi2_autograd(patches,order,ref,ref_inds,params):

    # -- unpack --
    t,c,h,w = params['shape']
    B,N,D = patches.shape
    # patches = rearrange(patches,'b n (t c h w) -> b n t c h w',t=t,c=c,h=h)
    # patches = rearrange(patches,'b n (t c h w) -> b n t c h w',t=t,c=c,h=h)

    # -- create weighted nn --
    model = WeightNet(B,N,D)
    wpatches,ave_wpatches = model(patches)


def gof_chi2_rand_subsample(data,order,ref,ref_inds,params):


    # -- unpack --
    t,c,h,w = params['shape']
    patches = rearrange(data,'b n (t c h w) -> b n t c h w',t=t,c=c,h=h)

    # # -- random subsampling --
    # for i in range(nrands):

