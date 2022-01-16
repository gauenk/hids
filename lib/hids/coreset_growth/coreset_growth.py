

# -- python --
import math

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- project --
from hids.l2_search import l2_search_masked,l2_search
from hids.utils import optional
# from hids.sobel import apply_sobel_to_patches
import hids.coreset_growth.log as log
from .utils import subset_data,subset_data_mask,update_mask,subset_at_curr_inds,subset_at_curr_inds_old
from .selection import select_ave_num,compute_subset_ave,filter_inds
from .growth_methods import gof_chi2_autograd,gof_chi2_rand_subsample,sharpness_subsample,exec_beam_search

def exec_coreset_growth(data,sigma,snum,**kwargs):

    # -- init subset --
    device = data.device
    bsize,num,dim = data.shape
    ave = th.zeros(bsize,1,dim,dtype=th.float32,device=device) # allocate once
    quants = th.ones(bsize,dtype=th.bool,device=device) # allocate once
    mask = th.zeros(bsize,num,dtype=th.int8,device=device)
    order = -th.ones(bsize,num,dtype=th.long,device=device)

    # -- init --
    ref_inds = -th.ones(bsize,num,device=device)
    ref_inds[:,0],ref_inds[:,1] = 0,1
    order = repeat(th.arange(num,device=device),'n -> b n',b=bsize)
    # quant = th.ones(bsize,dtype=th.bool,device=device) # allocate once

    # -- params --
    verbose = True
    max_ave = kwargs['nave']
    thresh = kwargs['thresh']
    step = kwargs['step']
    clean = optional(kwargs,'clean',None)
    pshape = optional(kwargs,'pshape',(2,3,7,7))
    method = optional(kwargs,'method',"chi2")
    if step == 0: thresh = 50.
    else: thresh = 1.
    param_log = log.create_params_log(**kwargs)
    if verbose: print(param_log)

    # -- method params --
    params = edict()
    params.shape = pshape

    # -- init data --
    cnum = 2
    mask[:,:cnum] = 1
    order[:,0],order[:,1] = 0,1
    ave[...] = th.mean(data[:,:cnum],1,keepdim=True)

    # -- run loop while references chage --
    params['rstep'] = 0
    delta_ref = 1.
    ref_inds_prev = None # init
    while delta_ref > 0: # usually a single pass, but "method" determines this
        ref_inds_prev = ref_inds.clone()
        update_order(data,sigma,ref,ref_inds,order)
        update_reference(data,order,ref,ref_inds,method,params)
        delta_ref = compute_delta_ref(ref_inds,ref_inds_prev)
        params['rstep'] += 1

    # -- setup return values for testing --
    a_data = gather_data(data,order)
    c_data = gather_data(clean,order)

    return order,a_data,c_data

def ref_inds2mask(inds):
    raise NotImplemented("")

def update_order(data,sigma,ref,ref_inds,order):

    # -- reference inds --
    ref_mask = ref_inds2mask(ref_inds)
    vals,inds = l2_search_masked(data,ref_mask,ref,sigma)

    # -- manage update [with mask] --
    order[...] = inds

def update_reference(data,order,ref,ref_inds,method,params):

    if method == "chi2_rand":
        return gof_chi2_rand_subsample(data,order,ref,ref_inds,params)
    elif method == "chi2_autograd":
        return gof_chi2_autograd(data,order,ref,ref_inds,params)
    elif method == "sharpness":
        return sharpness_subsample(data,order,ref,ref_inds,params)
    elif method == "beam":
        return exec_beam_search(data,order,ref,ref_inds,params)
    else:
        raise ValueError(f"Update reference [{method}]")

def compute_delta_ref(ref_inds,ref_inds_prev):
    return ((ref_inds - ref_inds_prev)**2).sum().item()

