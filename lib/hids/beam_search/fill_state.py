"""

Fill the proposed state according to selected sample

"""


# -- python --
import math
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- this package --
from hids.l2_search import l2_search
from hids.utils import optional

# -- local --
from .value_func import get_state_value_function
from .select import select_samples_to_search
from .state_utils import *
from .inds import *
from .init_state import init_prop_state

def fill_proposed_state(pstate,state,prop_samples,data,sigma,cnum):

    # -- shape --
    device = data.device
    D = data.shape[-1]
    B,nP,sW,max_num,dim = pstate.vecs.shape
    snum,nparticles = pstate.snum,pstate.nparticles

    # -- fill prop state with current state --
    init_prop_state(pstate,state,cnum)

    # -- "append" new indices @ cnum --
    pstate.inds[...,cnum] = prop_samples[...]

    # -- "append" next vectors @ cnum --
    # print("pre: ",pstate.vecs[0,0,:,cnum-1:cnum+1,0])
    for p in range(pstate.nparticles):
        aug_inds = repeat(prop_samples[:,p],'b n -> b n d',d=D)
        pstate.vecs[:,p,:,cnum,:] = th.gather(data,1,aug_inds)
    # print("post: ",pstate.vecs[0,0,:,cnum-1:cnum+1,0])
    # exit(0)

    # -- fill remaining vectors using newly computed ordering --
    fill_vecs_by_order(pstate,data,sigma,cnum)

def fill_vecs_by_order(pstate,data,sigma,cnum):

    # -- fill remaining vecs using ordering --
    bsize,num,dim = data.shape
    fnum = pstate.snum - cnum - 1
    for p in range(pstate.nparticles):
        for s in range(pstate.nsearch):

            # -- reference info --
            ref_num = get_ref_num(pstate,cnum)
            ref = th.mean(pstate.vecs[:,p,s,:cnum],1,keepdim=True)

            # -- compute order --
            ref_sigma = sigma / math.sqrt(ref_num)
            s_sigma = sigma**2 + ref_sigma**2
            compute_ordering(pstate,ref,data,s_sigma)

            # -- select remaining inds in order --
            rinds_ordered(pstate.inds[:,p,s,:cnum],pstate.order,
                          cnum,pstate.snum,pstate.remaining)
            # if p == 0:
            #     print(p,s,pstate.remaining[0])

            # -- append remaining inds in ordered --
            aug_remaining = repeat(pstate.remaining,'b n -> b n d',d=dim)
            pstate.vecs[:,p,s,cnum+1:,:] = th.gather(data,1,aug_remaining[:,:fnum])


