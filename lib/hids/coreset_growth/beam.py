"""

ToDo:

1. re-order the remaining based on current state
2. compute state value using the entire selected subset from current state

"""

# -- python --
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from hids.l2_search import l2_search
from hids.utils import optional
from .svf import get_state_value_function
from .state_utils import *

def beam_search(data,sigma,snum,bwidth=10,swidth=10,svf_method="svar",
                num_search=10,**kwargs):

    # -- shapes --
    verbose = True
    device = data.device
    bsize,num,dim = data.shape
    print("device: ",device)
    print("bsize,num,dim: ",bsize,num,dim)
    print("snum,bwidth,swidth,num_search: ",snum,bwidth,swidth,num_search)
    # print("-- info --")
    # print(data.max().item(),data.min().item(),sigma)

    # -- get state value function --
    sv_fxn = get_state_value_function(svf_method)
    sv_params = edict()
    sv_params.sigma = sigma
    sv_params.pshape = optional(kwargs,'pshape',(2,3,7,7))
    sv_params.edge_weight = optional(kwargs,'edge_weight',0.1)
    sv_params.max_mindex = optional(kwargs,'max_mindex',10)

    # -- create and init tensors --
    state,pstate = alloc_tensors(bsize,num,snum,dim,bwidth,swidth,device)
    cnum = init_state(state,data[:,None],sigma,0)
    init_state(pstate,data[:,None,None],sigma,0)

    # -- l2 search --
    l2_order = l2_search(data,data[:,[0]],sigma)[1]
    l2_order = optional(kwargs,'l2_order',l2_order)
    l2_order = repeat(l2_order,'b n -> b w n',w=bwidth)

    # -- exec search --
    if verbose: print("snum: ",snum)
    while cnum < snum:

        # -- create proposed state: S_t = S_{t-1} \cup {s'} --
        rnum = set_pstate(pstate,state,data,l2_order,sigma,cnum)
        assert rnum + cnum == num,"(rnum,cnum,num): (%d,%d,%d)" % (rnum,cnum,num)

        # -- compute value of states --
        sv_fxn(pstate.vals,pstate.vecs,cnum,sv_params)

        # -- keep best "bwidth" states --
        update_state(state,pstate,data,sigma,cnum)

        # -- update num --
        cnum += 1

        # -- terminate early --
        if verbose: print("[cnum/num_search]: %d/%d" % (cnum,num_search))
        if cnum >= num_search:
            terminate_early(state,data,l2_order,snum,cnum,sv_fxn,sv_params)
            break

    # -- pick final state using record --
    vals,inds,vecs = select_final_state(state)
    th.cuda.empty_cache()

    return vals,inds


def alloc_tensors(bsize,num,snum,dim,bwidth,swidth,device):
    """
    subset:
     vals: the value of each state
     -> "this allows us to pick the size of the subset"
     inds: the chosen indices

    state: the value, index of the top "bwidth" states

    pstate: proposed states to possibly replace state

    """

    # # -- subset --
    # subset = edict()
    # subset.vals = -th.ones(bsize,snum,dtype=th.float32,device=device)
    # subset.inds = -th.ones(bsize,snum,dtype=th.long,device=device)

    # -- subset --
    # db = edict()
    # db.vals = -th.ones(bsize,snum,dtype=th.float32,device=device)
    # db.inds = -th.ones(bsize,swidth,num,dtype=th.long,device=device)
    # db.vecs = -th.ones(bsize,swidth,num,dim,dtype=th.float32,device=device)

    # -- search state --
    state = edict()
    state.vals = -th.ones(bsize,bwidth,dtype=th.float32,device=device)
    state.inds = -th.ones(bsize,bwidth,snum,dtype=th.long,device=device)
    state.vecs = -th.ones(bsize,bwidth,snum,dim,dtype=th.float32,device=device)
    # state.order = th.zeros(bsize,bwidth,snum,dtype=th.long,device=device)
    # state.use_full = True

    # -- extra for indexing and selecting "snum" for each value --
    state.imask = th.zeros(bsize,bwidth,num,dtype=th.int8,device=device)

    # -- prop search state --
    pstate = edict()
    pstate.vals = -th.ones(bsize,bwidth,swidth,dtype=th.float32,device=device)
    pstate.inds = -th.ones(bsize,bwidth,swidth,snum,dtype=th.long,device=device)
    pstate.vecs = -th.ones(bsize,bwidth,swidth,snum,dim,dtype=th.float32,device=device)

    # -- alloc for later [no batching] --
    pstate.delta = th.zeros(bsize,num,dtype=th.float32,device=device)
    pstate.order = th.zeros(bsize,num,dtype=th.long,device=device)
    pstate.ref_type = "cnum"
    pstate.snum = snum
    # pstate.use_full = True

    return state,pstate

