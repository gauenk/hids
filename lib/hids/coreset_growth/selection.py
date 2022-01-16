

# -- python --
import torch as th
import numpy as np

# -- project --
from hids.sobel import apply_sobel_to_patches

def compute_subset_ave(data,mask,ave):
    mdata = data * mask[:,:,None]
    # print("data.shape: ",data.shape)
    # print("mask.shape: ",mask.shape)
    # print("mask.sum(1).shape: ",mask.sum(1).shape)
    # print(mdata.mean(1,keepdim=True).shape)
    # print(mask.sum(1)[:,None,None])
    ave[...] = mdata.sum(1,keepdim=True)/mask.sum(1)[:,None,None]

def filter_inds(inds,data,mask,ave,sigma,quants,max_ave,cnum,thresh,pshape):

    # -- create output --
    device = data.device
    bsize,num,dim = data.shape

    # -- edge case --
    mave = min(max_ave,cnum)
    if mave <= 2:
        ave[...] = th.mean(data[:,:mave],1,keepdim=True)
        return

    # -- select a number --
    pnum = cnum

    # -- compute ave based on variance of current size --
    # p_ave = th.mean(subset[:,:pnum],1,keepdim=True)

    # -- exec subset --
    c_quants = compute_edge_quant(data,mask,ave,pshape)
    # c_quants = compute_quant("edges",subset,p_ave,sigma,thresh,pshape)
    # c_quants = compute_quant(vals,thresh,"std")
    quants[...] = th.logical_and(c_quants,quants)
    # print("[cnum]: quant",cnum,quants.sum().item(),c_quants.sum().item())
    args = th.nonzero(quants)[:,0]
    print("args.shape: ",args.shape)
    # if len(args) == 0: return
    # aug_args = repeat(args,'b -> b 1 d',d=dim)
    # ave.scatter_(0,aug_args,p_ave)
    inds[args,0] = -1 # filter


def select_ave_num(subset,ave,sigma,quants,max_ave,cnum,thresh,pshape):
    """
    subset = B x N x D

    B = batchsize
    N = number of patches
    D = dim of each patch

    We want to take an average across N using a different
    number N_b of patches across the batch b in B.

    We pick N_b using the variance of the samples
    """

    # -- create output --
    device = subset.device
    bsize,num,dim = subset.shape

    # -- edge case --
    mave = min(max_ave,cnum)
    if mave <= 2:
        ave[...] = th.mean(subset[:,:mave],1,keepdim=True)
        return

    # -- select a number --
    pnum = cnum

    # -- compute ave based on variance of current size --
    p_ave = th.mean(subset[:,:pnum],1,keepdim=True)

    # -- exec subset --
    c_quants = compute_quant("edges",subset,p_ave,sigma,thresh,pshape)
    # c_quants = compute_quant(vals,thresh,"std")
    quants[...] = th.logical_and(c_quants,quants)
    # print("[cnum]: quant",cnum,quants.sum().item(),c_quants.sum().item())
    args = th.nonzero(quants)[:,0]
    if len(args) == 0: return
    aug_args = repeat(args,'b -> b 1 d',d=dim)
    ave.scatter_(0,aug_args,p_ave)


def compute_quant(method,subset,ave,sigma,thresh,pshape):
    if method == "std":
        vals,inds = l2_search(subset,ave,sigma)
        # print(vals.std(1)[:6].ravel())
        quant = vals.std(1) < thresh
        # print(quant[:6].ravel())
    elif method == "edges":
        quant = compute_edge_quant_subset(subset,ave,pshape)
    else:
        raise ValueError(f"Uknown method [{method}]")
    return quant


def compute_edge_quant(data,mask,ave,pshape):

    # -- sobel --
    ave_edges = apply_sobel_to_patches(ave,pshape)
    data_edges = apply_sobel_to_patches(data,pshape)

    # -- mask not in subset --
    # print("data_edges.shape: ",data_edges.shape)
    # mdata = data * mask[:,:,None]
    # print("mask.shape: ",mask.shape)

    # -- compute --
    lt = (ave_edges < data_edges)*mask
    quant = th.any(lt,1)

    return quant

def compute_edge_quant_subset(subset,ave,pshape):

    # -- sobel --
    ave_edges = apply_sobel_to_patches(ave,pshape)
    s_edges = apply_sobel_to_patches(subset,pshape)

    # -- compute if out --
    lt = ave_edges < s_edges
    quant = th.any(lt,1)
    return quant

