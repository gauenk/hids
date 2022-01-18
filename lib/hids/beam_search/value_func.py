"""
State value functions

"""

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from hids.utils import optional
from hids.sobel import apply_sobel_to_patches

def compute_state_value(pstate,sigma,cnum,sv_fxn,sv_params):
    sv_fxn(pstate.vals,pstate.vecs,cnum,sv_params)

def get_state_value_function(method):
    if method == "svar":
        return sample_var
    elif method == "svar_blur":
        return sample_var_blur
    else:
        raise ValueError(f"Update reference [{method}]")

def sample_var(vals,data,cnum,params):
    sigma = optional(params,'sigma',0.)
    # mindex = min(cnum,params.max_mindex)
    mean = data[:,:,:,:cnum].mean(-2,keepdim=True)
    vals[...] = ((data[:,:,:,:cnum] - mean)**2).mean((-2,-1)).pow(0.5)
    vals[...] = th.abs(vals - sigma) + mean.abs().mean((-2,-1))
    return vals

def sample_var_blur(vals,data,cnum,params):

    # -- sample variance --
    sample_var(vals,data,params)

    # -- [keep good edges] --
    B,W = data.shape[:2]
    mean = data.mean(3)
    data_rs = rearrange(mean,'b w s d -> (b w s) 1 d')
    edges = apply_sobel_to_patches(data_rs,params.pshape)
    edges = rearrange(edges,'(b w s) 1 -> b w s',b=B,w=W)
    vals[...] = vals[...] *(1 - params.edge_weight * th.abs(edges))

    return vals

