# -- python --
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from hids.l2_search import l2_search
from .svf import get_state_value_function

# ---------------------------------------
#
# -->       Manage "state"            <--
#
# ---------------------------------------

def terminate_early(state,data,l2_order,snum,cnum,sv_fxn,sv_params):
    """
    Desc: Add all remaining indices from the original l2
    ordering and compare results across particles

    """
    # -- shapes --
    device = state.imask.device
    bsize,nparticles = state.imask.shape[:2]
    bsize,num,dim = data.shape

    # -- init vec --
    fnum = snum - cnum

    # -- compute remaining inds --
    remain = remaining_ordered_inds(state,l2_order,cnum)

    # -- update state --
    for p in range(nparticles):

        # -- augment remaining inds --
        aug_remain = repeat(remain[:,p],'b n -> b n d',d=dim)

        # -- append to current state --
        state.inds[:,p,cnum:] = remain[:,p,:fnum]
        state.vecs[:,p,cnum:,:] = th.gather(data,1,aug_remain[:,:fnum])

    # -- update values --
    sv_fxn(state.vals[:,:,None],state.vecs[:,:,None],sv_params)

def update_state(state,pstate,cnum):
    """
    Desc: Set the current state according to the
    best "nsearch" from the proposed states

    """

    # -- shapes --
    BS,bW = state.vals.shape
    BS,bW,sW = pstate.vals.shape
    B,bW,sW,sN,D = pstate.vecs.shape

    # -- take topk --
    # pstate.vals
    # vals = rearrange(vals,'b w s -> b (w s)')
    inds = th.topk(pstate.vals,1,2,False).indices
    aug1_inds = repeat(inds,'b bw sw -> b bw sw n',n=sN)
    aug2_inds = repeat(aug1_inds,'b bw sw n -> b bw sw n d',d=D)

    # -- gather --
    state.vals[...] = th.gather(pstate.vals,2,inds)[:,:,0]
    state.inds[...] = th.gather(pstate.inds,2,aug1_inds)[:,:,0]
    state.vecs[...] = th.gather(pstate.vecs,2,aug2_inds)[:,:,0]

    # -- update mask from inds --
    state.imask[...] = 0
    state.imask[...].scatter_(2,state.inds[:,:,:cnum+1],1)

def topk_across_square():
    pass

def select_final_state(state):

    # -- shapes --
    B,W,N,D = state.vecs.shape

    # -- take topk --
    top1_inds = th.topk(state.vals,1,1,False).indices
    top1_aug1_inds = repeat(top1_inds,'b w -> b w n',n=N)
    top1_aug2_inds = repeat(top1_aug1_inds,'b w n -> b w n d',d=D)

    # -- gather --
    vals = th.gather(state.vals,1,top1_inds)[:,0]
    inds = th.gather(state.inds,1,top1_aug1_inds)[:,0]
    vecs = th.gather(state.vecs,1,top1_aug2_inds)[:,0]

    return vals,inds,vecs

# ---------------------------------------
#
# -->    Compute Remaining Indices    <--
#
# ---------------------------------------

def set_pstate(pstate,state,data,order,cnum):

    # -- shape --
    D = data.shape[-1]
    B,nP,sW,max_num,dim = pstate.vecs.shape

    # -- find remaining indices for search --
    remain = remaining_ordered_inds(state,order,cnum)
    rnum = remain.shape[-1]
    # info: remain.shape = (batch,nparticles,num-cnum)

    # -- select remaining indices for search --
    remain = select_prop_inds(remain,sW,cnum)
    aug_remain = repeat(remain,'b s n -> b s n d',d=D)
    # info: remain.shape = (batch,nparticles,nsearch)

    # -- create the proposed state --
    nparticles = nP
    for p in range(nparticles):

        # -- copy current state --
        pstate.vals[:,p,:] = state.vals[:,p,None]
        pstate.vecs[:,p,:,:cnum,:] = state.vecs[:,p,None,:cnum,:]
        pstate.inds[:,p,:,:cnum] = state.inds[:,p,None,:cnum]

        # -- "append" next state @ cnum --
        pstate.vecs[:,p,:,cnum,:] = th.gather(data,1,aug_remain[:,p])
        pstate.inds[:,p,:,cnum] = remain[:,p,:]#th.gather(,1,inds[:,p])

    return rnum

# ---------------------------------------
#
# -->    Compute Remaining Indices    <--
#
# ---------------------------------------

def remaining_ordered_inds(state,order,cnum):

    # -- shapes --
    device = state.vecs.device
    bsize,bwidth,snum,dim = state.vecs.shape
    nparticles = bwidth # rename
    bsize,num = order.shape

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
        imask[...] = th.gather(imask,1,order) # reorder using "order"

        # -- compute indices of orders to keep --
        nz = th.nonzero(imask)[:,1]
        nz = rearrange(nz,'(b n) -> b n',b=bsize)

        # -- gather orders to keep in order --
        remain[:,p] = th.gather(order,1,nz)[:,:rnum]
    return remain

def init_state(state,data,idx):
    state.vals[...,idx] = 0.
    state.inds[...,idx] = 0
    if 'imask' in state: state.imask[:,:,idx] = 1
    state.vecs[...,idx,:] = data[...,idx,:]
    return idx+1

def remaining_inds(mask):
    b,w,n = mask.shape
    inds = th.nonzero(mask < 0.5)[:,2]
    inds = rearrange(inds,'(b w n) -> b w n',b=b,w=w)
    return inds

# -------------------------------------
#
# -->       Select Prop.            <--
#
# -------------------------------------

def select_prop_inds(rinds,sW,cnum):
    if cnum == 1:
        return strided_inds(rinds,sW)
    else:
        # return rinds[:,:,:sW]
        return select_random_inds(rinds,sW)

def select_random_inds(rinds,sW):

    # -- create strided indices --
    device = rinds.device
    bsize,np,rnum = rinds.shape
    sinds = create_random_inds(bsize,np,sW,device,rnum)

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

