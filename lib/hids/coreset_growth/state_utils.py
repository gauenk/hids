# -- python --
import math
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
    sv_fxn(state.vals[:,:,None],state.vecs[:,:,None],snum,sv_params)

def update_state(state,pstate,data,sigma,cnum,use_full=True):
    """
    Desc: Set the current state according to the
    best "nsearch" from the proposed states

    """

    # -- shapes --
    BS,bW = state.vals.shape
    BS,bW,sW = pstate.vals.shape
    B,bW,sW,sN,D = pstate.vecs.shape

    # -- flatten proposed state --
    vals = rearrange(pstate.vals,'b w s -> b (w s)')
    inds = rearrange(pstate.inds,'b w s n -> b (w s) n')
    vecs = rearrange(pstate.vecs,'b w s n d -> b (w s) n d')
    # order = rearrange(pstate.order,'b w s n -> b (w s) n')
    # print("order.shape: ",order.shape)

    # -- take topk --
    inds_topk = th.topk(vals,bW,1,False).indices
    aug1_inds = repeat(inds_topk,'b k -> b k n',n=sN)
    aug2_inds = repeat(aug1_inds,'b k n -> b k n d',d=D)
    # aug3_inds = repeat(inds_topk,'b k -> b k n',n=N)
    # print("aug3_inds.shape: ",aug3_inds.shape)

    # -- gather --
    state.vals[...] = th.gather(vals,1,inds_topk)
    state.inds[...] = th.gather(inds,1,aug1_inds)
    state.vecs[...] = th.gather(vecs,1,aug2_inds)
    # state.order[...] = th.gather(order,1,aug3_inds)

    # -- update mask from inds --
    state.imask[...] = 0
    state.imask[...].scatter_(2,state.inds[:,:,:cnum+1],1)

def update_state_order(state,data,sigma,cnum):
    """
    Compute the ordering of the remaining indices
    and use them to fill the remaining vects.

    Note: We require using "proposed" state dimensions.
    """

    # -- expand if needed --
    # vecs = state.vecs
    # ndim = state.vecs.dim()
    # print("vecs.shape: ",vecs.shape)
    # if ndim == 4: vecs = vecs[:,:,None]

    # -- shapes --
    device = data.device
    bsize,num,dim = data.shape
    bsize,nparticles,nsearch = state.vecs.shape[:3]
    snum = state.snum
    rnum = state.snum - cnum

    # -- info --
    delta = th.abs(state.vecs[0,0,0,-1,:] - state.vecs[0,1,0,-1,:])
    delta = delta.sum(-1).mean().item()
    # print("delta: ",delta)
    # print(state.vecs[0,0,0,-1,:3])
    # print(state.vecs[0,1,0,-1,:3])
    # print(state.vecs[0,2,0,-1,:3])
    # print(state.vecs[0,0,1,-1,:3])
    # print(state.vecs[0,0,2,-1,:3])

    # -- compute ref info --
    rindex = get_ref_index(state,cnum)
    ref = th.mean(state.vecs[:,:,:,:rindex],3,keepdim=True)
    r_sigma = sigma / math.sqrt(cnum) # assume iid
    s_sigma = sigma**2 + r_sigma**2

    # -- fill [vec] values --
    for p in range(nparticles):
        for s in range(nsearch):

            # -- compute ordering [using alloced mem] --
            state.delta[...] = ((data - ref[:,p,s])**2).mean(2)
            state.delta[...] = th.abs(state.delta - s_sigma)
            state.order[...] = th.argsort(state.delta,1)
            # print(state.order[0,:3],state.order[0,-3:])

            # -- remove existing inds --
            remain = rinds_ordered(state.inds[:,p,s],state.order,cnum,snum,device)

            # -- set the remaining vectors --
            aug_order = repeat(state.order,'b n -> b n d',d=dim)
            state.vecs[:,p,s,cnum:,:] = th.gather(data,1,aug_order)[:,:rnum,:]

    # print("-"*30)
    # print(state.vecs[0,0,0,0,:3])
    # print(state.vecs[0,1,0,0,:3])
    # print(state.vecs[0,2,0,0,:3])
    # print(state.vecs[0,0,1,0,:3])
    # print(state.vecs[0,0,2,0,:3])

def get_ref_index(state,cnum):
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

def set_pstate(pstate,state,data,order,sigma,cnum,use_full=True):

    # -- shape --
    D = data.shape[-1]
    B,nP,sW,max_num,dim = pstate.vecs.shape

    # -- find remaining indices for search --
    # order = state.order if state.use_full else order
    remain = remaining_ordered_inds(state,order,cnum)
    rnum = remain.shape[-1]
    # [info]: remain.shape = (batch,nparticles,num-cnum)

    # -- select remaining indices for search --
    remain = select_prop_inds(remain,sW,cnum)
    aug_remain = repeat(remain,'b s n -> b s n d',d=D)
    # [info]: remain.shape = (batch,nparticles,nsearch)

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

    # -- update state ordering --
    update_state_order(pstate,data,sigma,cnum+1)
    # if use_full:
    #     print("data.shape: ",data.shape)
    #     update_state_order(pstate,data,sigma,cnum)
    #     add_remain_to_vecs(pstate,data,cnum)

    return rnum

# ---------------------------------------
#
# -->    Compute Remaining Indices    <--
#
# ---------------------------------------

def rinds_ordered(inds,order,cnum,snum,device):
    """
    a non-batched (across particles) version for updating
    the "vectors" using the newly computed ordering.

    used in "update_state_order"

    this is kind of a repeat of "remaining_ordered_inds" and
    should probably be merged.
    """

    # -- shapes --
    bsize,num = order.shape

    # -- allocate --
    rnum = num - cnum
    remain = th.zeros(bsize,rnum,dtype=th.long,device=device)
    imask = th.zeros(bsize,num,dtype=th.int8,device=device)

    # -- remove orders with mask --
    imask[...] = 1
    imask.scatter_(1,inds[:,:cnum],0) # remove already included
    imask[...] = th.gather(imask,1,order) # reorder using "order"

    # -- compute indices of orders to keep --
    nz = th.nonzero(imask)[:,1]
    nz = rearrange(nz,'(b n) -> b n',b=bsize)

    # -- gather orders to keep in order --
    remain = th.gather(order,1,nz)[:,:rnum]

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
    if 'order' in state:
        update_state_order(state,data.squeeze(),sigma,1)

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

