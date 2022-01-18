
import math
import torch as th

def l2_search_masked(data,mask,ref,sigma,sigma_ref=None,force=False):

    # -- sum sigma --
    if sigma_ref is None: sigma_ref = sigma
    s_sigma2 = sigma**2 + sigma_ref**2

    # -- compute l2 --
    vals = (data - ref)**2
    vals = th.abs(vals - s_sigma2)
    vals = th.mean(vals,2)

    # -- remove if mask --
    remove = th.nonzero(mask == 1,as_tuple=True)
    vals[remove] = float("inf")

    # -- ave and select --
    inds = th.argsort(vals,1)
    if force: force_zero_at_zero(inds)

    return vals,inds

def l2_search(data,ref,sigma,sigma_ref=None,force=False):

    # -- sum sigma --
    if sigma_ref is None: sigma_ref = sigma
    s_sigma2 = sigma**2 + sigma_ref**2

    # -- compute l2 --
    vals = (data - ref)**2
    vals = th.abs(vals - s_sigma2)
    vals = th.mean(vals,2)
    inds = th.argsort(vals,1)
    if force: force_zero_at_zero(inds)
    return vals,inds

def force_zero_at_zero(inds):
    pass

def l2_subset(data,sigma,snum):

    # -- subset the l2 search --
    vals,inds = l2_search(data,data[:,[0]],sigma)
    vals = th.gather(vals,1,inds[:,:snum])
    inds = inds[:,:snum]

    return vals,inds


