"""

Manage indices

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
from .value_func import get_state_value_function



def rinds_ordered(inds,order,_cnum,snum,remain=None):
    """
    a non-batched (across particles) version for updating
    the "vectors" using the newly computed ordering.

    used in "update_state_order"

    this is kind of a repeat of "remaining_ordered_inds" and
    should probably be merged.
    """

    # -- asserts --
    device = inds.device
    cnum = inds.shape[1]
    assert cnum == _cnum

    # -- allocate --
    bsize,num = order.shape
    rnum = num - cnum
    fnum = snum - cnum
    imask = th.zeros(bsize,num,dtype=th.int8,device=device)
    if remain is None:
        remain = th.zeros(bsize,fnum,dtype=th.long,device=device)

    # -- remove orders with mask --
    imask[...] = 1
    imask.scatter_(1,inds[:,:cnum],0) # remove already included
    imask[...] = th.gather(imask,1,order) # reorder using "order"

    # -- compute indices of orders to keep --
    nz = th.nonzero(imask[...])[:,1]
    nz = rearrange(nz,'(b n) -> b n',b=bsize)

    # -- gather orders to keep in order --
    remain[...] = th.gather(order,1,nz)[:,:remain.shape[1]]

    return remain


def remaining_ordered_inds(state,order,cnum):

    # -- shapes --
    device = state.vecs.device
    bsize,bwidth,snum,dim = state.vecs.shape
    nparticles = bwidth # rename
    bsize,nparticles,num = order.shape

    # -- unpack --
    inds = state.inds

    # -- allocate --
    rnum = num - cnum
    remain = th.zeros(bsize,nparticles,rnum,dtype=th.long,device=device)
    imask = th.zeros(bsize,num,dtype=th.int8,device=device)

    for p in range(nparticles):

        # -- remove orders with mask --
        imask[...] = 1
        imask.scatter_(1,inds[:,p,:cnum],0) # remove already included
        imask[...] = th.gather(imask,1,order[:,p]) # reorder using "order"

        # -- compute indices of orders to keep --
        nz = th.nonzero(imask)[:,1]
        nz = rearrange(nz,'(b n) -> b n',b=bsize)

        # -- gather orders to keep in order --
        remain[:,p] = th.gather(order[:,p],1,nz)[:,:rnum]
    return remain

def remaining_inds(mask):
    b,w,n = mask.shape
    inds = th.nonzero(mask < 0.5)[:,2]
    inds = rearrange(inds,'(b w n) -> b w n',b=b,w=w)
    return inds
