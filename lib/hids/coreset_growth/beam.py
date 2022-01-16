
# -- python --
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from .svf import get_state_value_function

def beam_search(data,sigma,snum,bwidth=10,swidth=None,svf_method="svar",**kwargs):

    # -- shapes --
    device = data.device
    bsize,num,dim = data.shape
    if swidth is None: swidth = 20#num

    # -- get state value function --
    sv_fxn = get_state_value_function(svf_method)
    sv_params = edict()
    sv_params.sigma = sigma

    # -- create and init tensors --
    state,pstate = alloc_tensors(bsize,num,snum,dim,bwidth,swidth,device)
    cnum = init_state(state,data[:,None],0)
    init_state(pstate,data[:,None,None],0)

    # -- exec search --
    print("snum: ",snum)
    while cnum < snum:

        # -- create proposed state: S_t = S_{t-1} \cup {s'} --
        rnum = set_pstate_vecs(pstate,state,data,cnum)
        assert rnum + cnum == num,"(rnum,cnum,num): (%d,%d,%d)" % (rnum,cnum,num)

        # -- compute value of states --
        pstate.vals[...] = sv_fxn(pstate.vecs[...,:cnum+1,:],sv_params)

        # -- keep best "bwidth" states --
        update_state(state,pstate,cnum)

        # -- update num --
        cnum += 1

    # -- pick final state using record --
    vals,inds,vecs = select_final_state(state)

    return vals,inds,vecs


def update_state(state,pstate,cnum):

    # -- shapes --
    BS,bW = state.vals.shape
    BS,bW,sW = pstate.vals.shape
    B,bW,sW,sN,D = pstate.vecs.shape

    # -- take topk --
    inds = th.topk(pstate.vals,1,2,False).indices
    # print("inds.shape: ",inds.shape,bW,sW)
    # print("pstate.vals.shape: ",pstate.vals.shape)
    # print("pstate.inds.shape: ",pstate.inds.shape)
    # print("pstate.vecs.shape: ",pstate.vecs.shape)
    # print("state.vals.shape: ",state.vals.shape)
    # print("state.inds.shape: ",state.inds.shape)
    # print("state.vecs.shape: ",state.vecs.shape)
    aug1_inds = repeat(inds,'b bw sw -> b bw sw n',n=sN)
    aug2_inds = repeat(aug1_inds,'b bw sw n -> b bw sw n d',d=D)

    # -- gather --
    state.vals[...] = th.gather(pstate.vals,2,inds)[:,:,0]
    state.inds[:,:,:] = th.gather(pstate.inds,2,aug1_inds)[:,:,0]
    state.vecs[:,:,:] = th.gather(pstate.vecs,2,aug2_inds)[:,:,0]
    # print(state.imask.shape,state.inds.shape)

    # print(state.imask[:,:].sum(2))
    # print(state.imask[:,:].sum(2).shape)
    # print("state.inds[:,:,:cnum+1].shape: ",state.inds[:,:,:cnum+1].shape)
    # print(state.inds[0,:,:cnum+1])

    state.imask[...] = 0
    state.imask[...].scatter_(2,state.inds[:,:,:cnum+1],1)

    check = state.imask[:,:].sum(2)
    check_v = check[0,0].item()
    print("check_v: ",check_v)
    nz_inds = th.nonzero(check != check_v)
    print("pstate.inds.shape: ",pstate.inds.shape)
    print("pstate.inds[18,0,:,:cnum+2]: ",pstate.inds[18,0,:,:cnum+2])
    print("state.inds: ",state.inds[18,:,:cnum+2])
    print("inds: ",inds[18].ravel())
    if len(nz_inds) > 0:
        print(nz_inds)
        bidx = nz_inds[0,0]
        print(state.inds[bidx,:,:cnum+1])
    assert th.all(check == check_v).item() is True
    print("checked: ",state.imask[0,:].sum(1))
    # print(state.imask[:,0].sum(1))

def select_final_state(state):
    # -- take topk --
    inds = th.topk(state.vals,1,1,False).indices[:,0]
    return state.vals[:,ind],state.inds[:,ind],state.vecs[:,ind]

def init_state(state,data,idx):
    state.vals[...,idx] = 0.
    state.inds[...,idx] = 0
    if 'imask' in state: state.imask[:,:,idx] = 1
    state.vecs[...,idx,:] = data[...,idx,:]
    return idx+1

def inverse_inds(mask):
    b,w,n = mask.shape
    # print(mask[0,:].sum(1),mask[0,:].max(1))
    inds = th.nonzero(mask < 0.5)[:,2]
    # print(inds.shape,b,w,n)
    inds = rearrange(inds,'(b w n) -> b w n',b=b,w=w)
    unum = None
    for p in range(inds.shape[1]):
        for b in range(inds.shape[0]):
            # print(b,p,len(inds[b,p].unique()))
            if unum is None:
                unum = len(inds[b,p].unique())
            else:
                continue
    # print("inds.shape: ",inds.shape)
    return inds

def set_pstate_vecs(pstate,state,data,cnum):

    # -- shape --
    B,nP,sW,max_num,dim = pstate.vecs.shape

    # -- create inds --
    inds = inverse_inds(state.imask)
    rnum = inds.shape[-1]

    # -- augment for "dim" --
    D = data.shape[-1]
    inds = inds[:,:,:sW]
    aug_inds = repeat(inds,'b s n -> b s n d',d=D)

    # -- "append" (of fill) a new vector @ cnum --
    nparticles = nP
    for p in range(nparticles):
        pstate.vecs[:,p,:,cnum,:] = th.gather(data,1,aug_inds[:,p])
        pstate.inds[:,p,:,cnum] = inds[:,p,:]#th.gather(,1,inds[:,p])
        # NOTE: this is where "duplicates" get added right now.
        # the seq "state.imask -> inds -> pstate.inds[@cnum]"
        # implies that "pstate.inds" updates ONLY at cnum
        # rather than the entire vector.

    return rnum

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

    # -- extra for indexing and selecting "snum" for each value --
    state.imask = th.zeros(bsize,bwidth,num,dtype=th.int8,device=device)


    # -- prop search state --
    pstate = edict()
    pstate.vals = -th.ones(bsize,bwidth,swidth,dtype=th.float32,device=device)
    pstate.inds = -th.ones(bsize,bwidth,swidth,snum,dtype=th.long,device=device)
    pstate.vecs = -th.ones(bsize,bwidth,swidth,snum,dim,dtype=th.float32,device=device)

    return state,pstate

