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
    for p in range(pstate.nparticles):
        aug_inds = repeat(prop_samples[:,p],'b n -> b n d',d=D)
        pstate.vecs[:,p,:,cnum,:] = th.gather(data,1,aug_inds)

    # -- fill remaining vectors using newly computed ordering --
    fill_vecs_by_order(pstate,data,sigma,cnum)

def fill_vecs_by_order(pstate,data,sigma,cnum):

    # -- fill remaining vecs using ordering --
    bsize,num,dim = data.shape
    fnum = pstate.snum - cnum
    for p in range(pstate.nparticles):
        for s in range(pstate.nsearch):

            # -- reference info --
            ref_num = get_ref_num(pstate,cnum)
            ref = th.mean(pstate.vecs[:,p,s,:cnum])

            # -- compute order --
            ref_sigma = sigma / math.sqrt(ref_num)
            s_sigma = sigma**2 + ref_sigma**2
            compute_ordering(pstate,ref,data,s_sigma)

            # -- select remaining inds in order --
            rinds_ordered(pstate.inds[:,p,s,:cnum],pstate.order,
                          cnum,pstate.snum,pstate.remaining)

            # -- append remaining inds in ordered --
            aug_remaining = repeat(pstate.remaining,'b n -> b n d',d=dim)
            pstate.vecs[:,p,s,cnum:,:] = th.gather(data,1,aug_remaining[:,:fnum])

def set_pstate(pstate,state,data,sigma,cnum,use_full=True):

    # -- shape --
    device = data.device
    D = data.shape[-1]
    B,nP,sW,max_num,dim = pstate.vecs.shape
    snum = pstate.snum

    # -- compute refs --
    rindex = get_ref_index(state,cnum)
    refs = th.mean(state.vecs[:,:,:,:rindex],3,keepdim=True)

    # -- create the proposed state --
    nparticles = nP
    for p in range(nparticles):
        for s in range(nsearch):

            # -- compute ordering [using alloced mem] --
            pstate.delta[...] = ((data - refs[:,p,s])**2).mean(2)
            pstate.delta[...] = th.abs(pstate.delta - s_sigma)
            pstate.order[...] = th.argsort(pstate.delta,1)
            order = pstate.order

            # -- remove existing inds --
            remain = rinds_ordered(state.inds[:,p,s],order,cnum+1,snum,device)

            # -- set the remaining vectors --
            aug_order = repeat(state.order,'b n -> b n d',d=dim)
            state.vecs[:,p,s,cnum+1:,:] = th.gather(data,1,aug_order)[:,:rnum+1,:]

            # -- find remaining indices for search --
            # order = state.order if state.use_full else order
            # remain = remaining_ordered_inds(state,order,cnum)
            # rnum = remain.shape[-1]
            # [info]: remain.shape = (batch,nparticles,num-cnum)

            # -- select remaining indices for search --
            remain = select_prop_inds(remain,sW,cnum)
            aug_remain = repeat(remain,'b s n -> b s n d',d=D)
            # [info]: remain.shape = (batch,nparticles,nsearch)

            # -- copy current state --
            pstate.vals[:,p,:] = state.vals[:,p,None]
            pstate.vecs[:,p,:,:cnum,:] = state.vecs[:,p,None,:cnum,:]
            pstate.inds[:,p,:,:cnum] = state.inds[:,p,None,:cnum]

            # -- "append" next state @ cnum --
            pstate.vecs[:,p,:,cnum,:] = th.gather(data,1,aug_remain[:,p])
            pstate.inds[:,p,:,cnum] = remain[:,p,:]#th.gather(,1,inds[:,p])

    # -- update state ordering --
    # update_state_order(pstate,data,sigma,cnum+1)
    # if use_full:
    #     print("data.shape: ",data.shape)
    #     update_state_order(pstate,data,sigma,cnum)
    #     add_remain_to_vecs(pstate,data,cnum)

    return rnum


