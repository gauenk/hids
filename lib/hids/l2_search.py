
import torch as th

def l2_search_masked(data,mask,ref,sigma,force=False):

    # -- compute l2 --
    vals = (data - ref)**2
    vals = th.abs(vals - 2*sigma)
    vals = th.mean(vals,2)

    # -- remove if mask --
    print("vals.shape: ",vals.shape)
    print("mask.shape: ",mask.shape)
    remove = th.nonzero(mask == 1,as_tuple=True)
    print(remove)
    # print("remove.shape: ",remove.shape)
    # print(remove[:10])
    # inf_gpu = th.FloatTensor([float("inf")]).to(vals.device)
    vals[remove] = 1000#inf_gpu#float("inf")
    print(vals[0,10])

    # -- ave and select --
    inds = th.argsort(vals,1)
    if force: force_zero_at_zero(inds)

    return vals,inds

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


