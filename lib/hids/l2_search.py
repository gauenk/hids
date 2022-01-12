
import torch as th

def l2_search(data,ref,sigma,force=False):

    # -- compute l2 --
    vals = (data - ref)**2
    vals = th.abs(vals - 2*sigma)
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


