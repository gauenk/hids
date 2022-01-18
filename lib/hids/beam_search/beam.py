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

# -- this package --
from hids.l2_search import l2_search
from hids.utils import optional

# -- local --
from .value_func import get_state_value_function,compute_state_value
from .state_utils import *
from .propose import propose_state
from .update import update_state,terminate_early,select_final_state
from .init_state import init_state_pair

def beam_search(data,sigma,snum,bwidth=10,swidth=10,svf_method="svar",
                num_search=10,**kwargs):

    # -- shapes --
    verbose = False
    device = data.device
    bsize,num,dim = data.shape

    # -- get state value function --
    sv_fxn = get_state_value_function(svf_method)
    sv_params = edict()
    sv_params.sigma = sigma
    sv_params.pshape = optional(kwargs,'pshape',(2,3,7,7))
    sv_params.edge_weight = optional(kwargs,'edge_weight',0.1)
    sv_params.max_mindex = optional(kwargs,'max_mindex',10)

    # -- create and init tensors --
    state,pstate = init_state_pair(data,sigma,bsize,num,snum,dim,bwidth,swidth,device)
    cnum = 1

    # -- exec search --
    if verbose: print("snum: ",snum)
    while cnum < snum:

        # -- propose state --
        propose_state(state,pstate,data,sigma,cnum)
        compute_state_value(pstate,sigma,cnum,sv_fxn,sv_params)
        update_state(state,pstate,data,sigma,cnum)
        # print(state.vals[0])

        # -- update num --
        cnum += 1

        # -- terminate early --
        if verbose: print("[cnum/num_search]: %d/%d" % (cnum,num_search))
        if cnum >= num_search:
            terminate_early(state,data,sigma,snum,cnum,sv_fxn,sv_params)
            break

    # -- pick final state using record --
    vals,inds,vecs = select_final_state(state)
    th.cuda.empty_cache()

    return vals,inds


