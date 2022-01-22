"""
State value functions

"""

# -- python --
import math

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- local --
from hids.utils import optional
from hids.sobel import apply_sobel_to_patches
from hids.patch_utils import denoise_subset

def compute_state_value(pstate,sigma,cnum,sv_fxn,sv_params):
    sv_fxn(pstate.vals,pstate.vecs,cnum+1,sv_params)

def get_state_value_function(method):
    if method == "svar":
        return sample_var
    elif method == "svar_blur":
        return sample_var_blur
    else:
        raise ValueError(f"Update reference [{method}]")

def sample_var(vals,data,cnum,params):

    # -- mean indexing --
    # mindex = min(cnum,params.max_mindex)
    mindex = min(cnum,5)#params.max_mindex)
    # mindex = cnum

    # -- noise level --
    sigma = optional(params,'sigma',0.)
    t_sigma = compute_target_sigma(sigma,mindex)

    # mean = data[:,:,:,:mindex].mean(-2,keepdim=True)
    # # mean = data[:,:,:,:].mean(-2,keepdim=True)

    # data_zm = data[:,:,:,:cnum,:] - mean
    # mean_error = 0.#th.abs(data_zm.mean((-2,-1)))
    # C = cnum/(cnum-1.)
    # std_error = (data_zm**2).mean((-2,-1))*C - t_sigma
    # # std_error = data_zm.std((-2,-1)) - t_sigma
    # print(std_error[0,0],data_zm[0,0,0,0])
    # # zero if < zero
    # std_error[th.nonzero(std_error < 0.)] = 0.
    # vals[...] = mean_error + std_error

    # print("t_sigma: ",t_sigma)

    # -- loss v 1 --
    # mean = data[:,:,:,:].mean(-2,keepdim=True)
    # data_zm = data[:,:,:,:] - mean
    # tmp = ((data_zm)**2).mean(-1).pow(0.5).mean(-1)
    # vals[...] = tmp

    # vals[...] = (tmp - t_sigma).mean(-1)
    # vals[...] -= 1*(data_zm.std(-1).mean(-1) - t_sigma)

    # -- loss v 2 [(dim) ave std  - std of ave] --
    # term_1 = data[...,:cnum].mean(-2).std(-1)
    # term_2 = data[...,:cnum].std(-1).mean(-1)
    # vals[...] = th.abs(term_1 - term_2)

    # ------------------
    #
    # -->  loss v 3  <--
    #
    # ------------------


    # -- [var term] --
    # mean = data[:,:,:,:mindex].mean(-2,keepdim=True)
    # mean = data[:,:,:,:mindex].mean(-2,keepdim=True)
    # data_zm = data[:,:,:,:cnum] - mean
    # tmp = ((data_zm)**2).mean(-1).pow(0.5)
    # vals[...] = (tmp - t_sigma).mean(-1)

    # -- [mean term] --
    # if mindex == cnum: mindex = 1
    if cnum >= 100 and False:
        vals[...] = ((data[...,:2,:].mean(-2,keepdim=True) - data[...,:,:])**2).mean((-2,-1))
    elif False:
        mean = data[...,:,:].mean(-2,keepdim=True)
        data_zm = data[:,:,:,:] - mean
        vals[...] = keep_meaned_data_different(data_zm,cnum,-1)#mindex)
    else:
        # print("data.shape: ",data.shape)
        b,w,s = data.shape[:3]
        # data = data[...,:,:]
        # deno = data
        deno = data

        # sub = rearrange(data,'b bw s n p -> (b bw s) n p')
        # deno = denoise_subset(sub,sigma)
        # deno = rearrange(deno,'(b bw s) n t c h w -> b bw s n (t c h w)',b=b,bw=w)
        # print("-"*30)
        # print(deno[1,:,:,0,0])
        # print(deno[1,:,:,-1,0])
        # print(deno[1,:,:,0,1])
        # print(deno[1,:,:,-1,1])

        # -- some deno consistency --
        vals[...] = 0.
        mnum = cnum
        num,dim = data.shape[-2:]
        Z = num * dim
        for i in range(mnum):
            res = deno[...,[i],:] - data[...,:,:]
            # res = deno[...,[i],:] - deno[...,:,:]
            # vals[...] += th.abs(((res/sigma)**2).sum((-2,-1)) - Z)
            vals[...] += ((res)**2).mean((-2,-1))
        vals[...] /= mnum

        # -- another deno consistency --
        # dmean = deno[...,:cnum,:].mean(-2,keepdim=True)

        # vals[...] = ((deno[...,:cnum,:] - data[...,:cnum,:])**2).mean((-2,-1))

        # vals[...] = ((res.std((-2)) - sigma)**2).mean(-1)
        # vals[...] += ((res.std((-1)) - sigma)**2).mean(-1)
        # vals[...] = ((data[...,[0],:]- data.mean(-2,keepdim=True))**2).mean((-2,-1))
        # vals[...] = ((data[...,[0],:]- data)**2).mean((-2,-1))
        # vals[...] += (res**2).mean((-2,-1))# + (res.std((-2,-1)) - sigma)**2
        # print(vals[0])
        # print(vals[1])
        # print(deno[0,:,0,[0],0].ravel())
        # vals[...] = th.mean( ( deno[...,[0],:] - deno )**2 , (-2,-1))*10
        # denoise_patches(data,sigma)

    # vals[...] = th.abs(vals - t_sigma).mean(-1)
    # mean = data[:,:,:,:cnum].mean(-2,keepdim=True)
    # vals[...] = th.abs(vals - t_sigma)
    # vals[th.nonzero(vals<0.)] = 0.

    return vals

def keep_meaned_data_different(data,cnum,mindex):
    print("data.shape: ",data.shape,cnum,mindex)
    # deltas = data[...,None,:cnum,:] - data[...,:mindex,None,:]
    deltas = data[...,:cnum,None,:] - data[...,:cnum,:,None]
    print("deltas.shape: ",deltas.shape)
    deltas = deltas**2
    deltas = svd_term(deltas)
    # deltas = th.exp(-deltas**2/2.)
    deltas = deltas.mean((-3,-2,-1))
    return deltas

# def svd_term(deltas):
#     b, w, s, n, p1, p2 = deltas.shape
#     deltas = rearrange(deltas,'b w s n p1 p2 -> (b w s n) p1 p2')
#     print("deltas.shape: ",deltas.shape)
#     th.linalg.svd(deltas

def compute_target_sigma(sigma,m):
    var = sigma**2
    t_sigma2 = ((m-1)/m)**2 * var + (m-1)/(m**2) * var
    t_sigma = math.sqrt(t_sigma2)
    return t_sigma

def sample_var_blur(vals,data,cnum,params):

    # -- sample variance --
    sample_var(vals,data,cnum,params)

    # -- [keep good edges] --
    B,W = data.shape[:2]
    mean = data.mean(3)
    data_rs = rearrange(mean,'b w s d -> (b w s) 1 d')
    edges = apply_sobel_to_patches(data_rs,params.pshape)
    edges = rearrange(edges,'(b w s) 1 -> b w s',b=B,w=W)
    vals[...] = vals[...] - params.edge_weight * th.abs(edges)

    return vals

