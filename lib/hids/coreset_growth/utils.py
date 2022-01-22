
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- project --
from hids.utils import gather_data
from hids.patch_utils import expand_patches
from hids.deno import denoise_patches

def compute_dist_l2(data,ref):
    dist = (data - ref)**2
    dist = th.mean(dist,2)
    return dist

def compute_dist(data,ref,method):
    if method == "l2":
        return compute_dist_l2(data,ref)
    else:
        raise ValueError(f"Dist method is [{method}]")

def subset_data_mask(data,mask):
    bsize,num,dim = data.shape
    inds = th.nonzero(mask)[:,1]
    inds = rearrange(inds,'(b n) -> b n',b=bsize)
    return gather_data(data,inds),inds

def subset_data(data,order,pick):
    if pick == "inc":
        return subset_data_inc(data,order)
    elif pick == "exc":
        return subset_data_exc(data,order)
    else:
        raise ValueError(f"subset data using [{pick}]")

def subset_data_exc(data,order):
    bsize,num,dim = data.shape
    bsize,num = order.shape
    inds = selected_order(order)
    mask = th.zeros(bsize,num,dtype=th.int8,device=data.device)
    mask = mask.scatter(1,inds,1.)
    check_mask_req(mask)
    return subset_data_mask(data,mask == 0)

def subset_data_inc(data,order):
    bsize,num,dim = data.shape
    bsize,num = order.shape
    inds = selected_order(order)
    return gather_data(data,inds)

def subset_mask(mask,b_mask):
    bsize,num = mask.shape
    inds = th.nonzero(b_mask)[:,1]
    inds = rearrange(inds,'(b n) -> b n',b=bsize)
    return th.gather(mask,1,inds)

def selected_order(order):
    bsize,num = order.shape
    args = th.nonzero(th.all(order>=0,0))[:,0]
    snum = args[-1].item()+1
    inds = order[:,:snum]
    return inds

def check_mask_req(mask):
    # check mask rows are equal
    rsum = mask.sum(1)
    snum = rsum[0].item()
    assert th.all(snum == rsum).item() is True

def update_mask(mask,inds):

    # -- unpack --
    bsize,num = mask.shape
    curr = mask.sum(1)

    # -- valid inds are not "-1" --
    vinds = th.nonzero(th.all(inds!=-1,1))[:,0]

    # -- update inds w.r.t zero mask --
    db_mask = subset_mask(mask,mask == 0) # to add
    db_inds = th.nonzero(mask == 0)[:,1]
    db_inds = rearrange(db_inds,'(b n) -> b n',b=bsize)
    # db_mask = th.scatter(db_mask,1,inds[:,[0]],1.)
    tmp = th.scatter(db_mask,1,inds,1.)
    db_mask[vinds] = tmp[vinds]
    # db_mask[vinds] = th.scatter(db_mask[vinds],1,inds[vinds],1.)
    # db_mask = th.scatter(db_mask,1,inds,1.)

    # -- update inds w.r.t full mask --
    mask[...] = th.scatter(mask,1,db_inds,db_mask)


    nchange = mask.sum(1) - curr
    return nchange

def denoise_subset(noisy,sigma):

    # -- expand patches --
    patches = expand_patches(noisy)

    # -- exec denoising --
    patches_p = patches.clone()
    denoise_patches(patches,sigma)
    deno = patches

    return deno

def subset_at_curr_inds(data,order,inds,db_inds,cnum,snum):
    """
    Get the subset using the current ave. ordering
    """
    # -- get relevant patches --
    nfill = snum - cnum
    order = order.clone()
    sel_inds = th.gather(db_inds,1,inds[:,:nfill])
    order[:,cnum:cnum+nfill] = sel_inds
    subset = subset_data(data,order,"inc")
    assert subset.shape[1] == snum
    return subset

def subset_at_curr_inds_old(data,vals,inds,mask,snum):
    """
    Get the subset using the current ave ordering
    """
    # -- get relevant patches --
    mask = mask.clone()
    nfill = snum - int(mask[0].sum().item())
    update_mask(mask,inds[:,:nfill])
    subset,_ = subset_data_mask(data,mask == 1)

    assert subset.shape[1] == snum
    return subset

