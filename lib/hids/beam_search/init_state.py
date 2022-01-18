
# -- python --
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def init_state_pair(data,sigma,bsize,num,snum,dim,bwidth,swidth,device):
    state,pstate = alloc_state(bsize,num,snum,dim,bwidth,swidth,device)
    cnum = init_state(state,data[:,None],sigma,0)
    init_state(pstate,data[:,None,None],sigma,0)
    return state,pstate

def alloc_state(bsize,num,snum,dim,bwidth,swidth,device):
    """
    subset:
     vals: the value of each state
     -> "this allows us to pick the size of the subset"
     inds: the chosen indices

    state: the value, index of the top "bwidth" states

    pstate: proposed states to possibly replace state

    """

    #
    # -- search state --
    #

    state = edict()
    state.vals = -th.ones(bsize,bwidth,dtype=th.float32,device=device)
    state.inds = -th.ones(bsize,bwidth,snum,dtype=th.long,device=device)
    state.vecs = -th.ones(bsize,bwidth,snum,dim,dtype=th.float32,device=device)
    # state.order = th.zeros(bsize,bwidth,snum,dtype=th.long,device=device)
    # state.use_full = True

    # -- extra for indexing and selecting "snum" for each value --
    state.imask = th.zeros(bsize,bwidth,num,dtype=th.int8,device=device)

    # -- alloc for later [no batching] --
    state.delta = th.zeros(bsize,num,dtype=th.float32,device=device)
    state.order = th.zeros(bsize,num,dtype=th.long,device=device)
    state.remaining = th.zeros(bsize,snum,dtype=th.long,device=device)

    # -- misc --
    state.ref_type = "cnum"
    state.snum = snum

    #
    # -- prop search state --
    #

    pstate = edict()
    pstate.vals = -th.ones(bsize,bwidth,swidth,dtype=th.float32,device=device)
    pstate.inds = -th.ones(bsize,bwidth,swidth,snum,dtype=th.long,device=device)
    pstate.vecs = -th.ones(bsize,bwidth,swidth,snum,dim,dtype=th.float32,device=device)

    # -- alloc for later [no batching] --
    pstate.delta = th.zeros(bsize,num,dtype=th.float32,device=device)
    pstate.order = th.zeros(bsize,num,dtype=th.long,device=device)
    pstate.remaining = th.zeros(bsize,snum,dtype=th.long,device=device)

    # -- misc --
    pstate.ref_type = "cnum"
    pstate.snum = snum
    pstate.nsearch = swidth
    pstate.swidth = swidth
    pstate.nparticles = bwidth
    # pstate.use_full = True

    return state,pstate


def init_state(state,data,sigma,idx):

    # -- assign zero index --
    state.vals[...,idx] = 0.
    state.inds[...,idx] = 0
    if 'imask' in state: state.imask[:,:,idx] = 1

    # -- fill vecs --
    shape = list(state.vecs.shape)
    shape[0],shape[-2],shape[-1] = -1,-1,-1
    state.vecs[...,idx,:] = data[...,idx,:]

    # -- update state ordering --
    # if 'order' in state:
    #     update_state_order(state,data.squeeze(),sigma,1)

    return idx+1


def init_prop_state(pstate,state,cnum):
    # -- fill across particles --
    for p in range(pstate.nparticles):
        # -- copy current state --
        pstate.vals[:,p,:] = state.vals[:,p,None]
        pstate.vecs[:,p,:,:cnum,:] = state.vecs[:,p,None,:cnum,:]
        pstate.inds[:,p,:,:cnum] = state.inds[:,p,None,:cnum]


