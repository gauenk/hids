"""

Utilities to manage states

"""

# -- python --
import math
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from hids.l2_search import l2_search
# from .value_func import get_state_value_function

def compute_ordering(state,ref,data,s_sigma):
    # -- compute ordering [using alloced mem] --
    state.delta[...] = ((data - ref)**2).mean(2)
    state.delta[...] = th.abs(state.delta - s_sigma)
    state.order[...] = th.argsort(state.delta,1)

def get_ref_num(state,cnum):
    if state.ref_type == "cnum":
        return cnum
    else:
        raise KeyError(f"Reference index not assigned [{cnum}]")

def add_remain_to_vecs(pstate,data,cnum):

    # -- shape --
    D = data.shape[-1]
    B,nP,sW,snum,dim = pstate.vecs.shape
    fnum = snum - cnum

    # -- compute remaining inds --
    remain = remaining_ordered_inds(pstate,pstate.order,cnum)
    rnum = remain.shape[-1]

    # -- add data to vector --
    for p in range(nparticles):
        pstate.vecs[:,p,cnum+1:,:] = torch.gather(data,1,remain[:,p,:fnum])


