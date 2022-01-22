
"""

Propose a new state to search

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
from .value_func import get_state_value_function
from .select import select_samples_to_search
from .fill_state import fill_proposed_state
from .state_utils import *
from .inds import *

# -----------------------------
#
# -->   Primary Functions   <--
#
# -----------------------------

def propose_state(state,pstate,data,sigma,cnum):

    # -- propose samples to search -
    candidates = get_candidate_samples(state,data,sigma,cnum)
    prop_samples = select_samples_to_search(candidates,pstate.nsearch,cnum)
    # should be (B,P,S)

    # -- fill proposal state with according to new samples --
    fill_proposed_state(pstate,state,prop_samples,data,sigma,cnum)

# -------------------------------------------------------
#
#  "Candidate Samples":
#  e.g. the ones not already included in the state
#
# ------------------------------------------------------

def get_candidate_samples(state,data,sigma,cnum):

    # -- shapes --
    device = data.device
    bsize,nparticles,snum,dim = state.vecs.shape
    bsize,num,dim = data.shape

    # -- alloc --
    candidates = th.zeros(bsize,nparticles,snum,dtype=th.long,device=device)

    # -- iterate --
    for p in range(nparticles):

        # -- reference info --
        ref_num = get_ref_num(state,cnum)
        ref = th.mean(state.vecs[:,p,:ref_num],1,keepdim=True)

        # -- compute order --
        ref_sigma = sigma / math.sqrt(ref_num)
        s_sigma = sigma**2 + ref_sigma**2
        compute_ordering(state,ref,data,s_sigma)
        # if p == 0:
        #     print("c")
        #     print(state.order[0,:])
        #     # print(state.delta[0,:])
        #     print(state.delta[0,387])
        #     print(state.delta[0,0])

        # -- order inds --
        rinds_ordered(state.inds[:,p,:cnum],state.order,
                      cnum,snum,candidates[:,p])

    return candidates

