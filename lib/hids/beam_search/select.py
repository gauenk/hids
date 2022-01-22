# -- python --
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- this package --
from hids.l2_search import l2_search
from hids.utils import optional

# -----------------------------
#
#  Selecting Samples to Search
#
# -----------------------------

def select_samples_to_search(remaining_inds,nsearch,cnum):
    """
    Select the samples to search from the candidate list
    """
    if cnum == 1:
        return strided_inds(remaining_inds,nsearch)
    else:
        return remaining_inds[:,:,:nsearch]
        # print(strided_inds(remaining_inds,nsearch)[0,:,0])
        # return strided_inds(remaining_inds,nsearch)
        # return select_random_inds(remaining_inds,nsearch)

def select_random_inds(rinds,sW):

    # -- create strided indices --
    device = rinds.device
    bsize,np,rnum = rinds.shape
    sinds = create_wrandom_inds(bsize,np,sW,device,rnum)
    # sinds = create_random_inds(bsize,np,sW,device,rnum)

    # -- gather remaining inds --
    strided_inds = th.gather(rinds,2,sinds)

    return strided_inds

def strided_inds(rinds,sW):

    # -- create strided indices --
    device = rinds.device
    bsize,np,rnum = rinds.shape
    sinds = create_strided_inds(bsize,np,sW,device,rnum)

    # -- gather remaining inds --
    strided_inds = th.gather(rinds,2,sinds)

    return strided_inds

def create_wrandom_inds(bsize,np,sW,device,rnum):
    # -- alloc --
    inds = th.zeros(bsize,np,sW,dtype=th.long,device=device)
    menu = 1./(th.arange(rnum)+1)

    # -- create blocks --
    for p in range(np):
        inds[:,p,:] = th.multinomial(menu,sW)

    return inds

def create_random_inds(bsize,np,sW,device,rnum):

    # -- alloc --
    inds = th.zeros(bsize,np,sW,dtype=th.long,device=device)

    # -- create blocks --
    for p in range(np):
        vec = th.randperm(rnum)[:sW]
        inds[:,p,:] = repeat(vec,'s -> b s',b=bsize)
    return inds

def create_strided_inds(bsize,np,sW,device,rnum):

    # -- alloc --
    inds = th.zeros(bsize,np,sW,dtype=th.long,device=device)

    # -- create blocks --
    for p in range(np):
        vec = th.remainder(th.arange(sW)+sW*p,rnum)
        inds[:,p,:] = repeat(vec,'s -> b s',b=bsize)
    return inds

