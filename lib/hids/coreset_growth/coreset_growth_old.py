

# -- python --
import math

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- project --
from hids.l2_search import l2_search_masked,l2_search
from hids.utils import optional
from hids.sobel import apply_sobel_to_patches
import hids.coreset_growth.log as log
from hids.patch_utils import expand_patches,denoise_patches
from .utils import subset_data,subset_data_mask,update_mask,subset_at_curr_inds,subset_at_curr_inds_old
from .selection import select_ave_num,compute_subset_ave,filter_inds

def exec_coreset_growth(data,sigma,snum,**kwargs):

    # -- init subset --
    device = data.device
    bsize,num,dim = data.shape
    # assert bsize == 1,"only one per batch"

    # -- init subset --
    # subset = th.zeros_like(bsize,snum,dim)
    # subset[:,0,:] = data[:,0]
    ave = th.zeros(bsize,1,dim,dtype=th.float32,device=device) # allocate once
    quants = th.ones(bsize,dtype=th.bool,device=device) # allocate once
    # ave_mask = th.zeros(bsize,num,dim,dtype=th.byte,device=device) # allocate once
    mask = th.zeros(bsize,num,dtype=th.int8,device=device)
    order = -th.ones(bsize,num,dtype=th.long,device=device)

    # -- run loop --
    verbose = True
    max_ave = kwargs['nave']
    thresh = kwargs['thresh']
    step = kwargs['step']
    clean = optional(kwargs,'clean',None)
    pshape = optional(kwargs,'pshape',(2,3,7,7))
    if step == 0: thresh = 50.
    else: thresh = 1.

    # -- init --
    cnum = 2
    order[:,0] = 0
    order[:,1] = 1
    # order[:,2] = 2
    mask[:,:cnum] = 1
    ave[...] = th.mean(data[:,:cnum],1,keepdim=True)

    # -- info --
    if verbose:
        print("-"*30)
        print("max_ave: ",max_ave)
        print("thresh: ",thresh)
        print("step: ",step)
        print("sigma: ",sigma)

    while cnum < snum:

        # -- split data --
        # db_m = subset_data_mask(data,mask == 0) # to add
        # subset_m = subset_data_mask(data,mask == 1) # in the subset
        db_inds = th.nonzero(mask == 0)
        # db,db_inds = subset_data(data,order,"exc") # to add
        # subset = subset_data(data,order,"inc") # in the subset
        # assert th.sum((db_m - db)**2).item() < 1e-8
        # assert th.sum((subset_m - subset)**2).item() < 1e-8
        # if not(clean is None):
        #     # subset_c = subset_data(clean,mask == 1)
        #     subset_c = subset_data(clean,order,"inc")

        # -- compute ref --
        # print("ave[:3,:2,0]: ",ave[:3,:2])
        if step == 0:
            if cnum <= max_ave:
                compute_subset_ave(data,mask,ave)
                # select_ave_num_old(subset,mask,ave,sigma,quants,
                #                    max_ave,cnum,thresh,pshape)
        else:
            anum = min(max_ave,cnum)
            compute_subset_ave(data,mask,ave)
            # ave[...] = th.mean(subset[:,:anum,:],1,keepdim=True)

        # print("ave[:3,:2,0]: ",ave[:3,:2])
        # ave = compute_subset_ave(subset,anums,ave_mask)
        # anum = min(max_ave,cnum)
        # ave = th.mean(subset[:,:anum,:],1,keepdim=True)

        # -- sobel --
        # pt,c,ph,pw = 2,3,7,7
        # ave_edges = apply_sobel_to_patches(ave,pshape)#pt,c,ph,pw)
        # s_edges = apply_sobel_to_patches(subset,pshape)#pt,c,ph,pw)

        # -- reorder --
        vals,inds = l2_search_masked(data,mask,ave,sigma)
        print("[cg] inds.shape: ",inds.shape)
        # vals,inds = l2_search(subset,ave,sigma)
        if verbose and cnum <= max_ave:#max_ave == cnum:
            print("-"*20)
            print("---- [cnum: %d] ----" % cnum)
            print("-"*20)
            edges_log = log.create_edges_log(data[:,[0]],ave,sigma,pshape,10,1)
            l2vals_log = log.create_l2vals_log(data[:,[0]],ave,sigma,10,1)
            # l2vals_log = log.create_l2vals_log(db,ave,sigma,10,5)

            # npatches = subset_at_curr_inds(data,order,inds,db_inds,cnum,snum)
            # cpatches = subset_at_curr_inds(clean,order,inds,db_inds,cnum,snum)
            # npatches = subset_at_curr_inds_old(data,vals,inds,mask,snum)
            # cpatches = subset_at_curr_inds_old(clean,vals,inds,mask,snum)

            # print("npatches.shape: ",npatches.shape)
            # cpatches = clean[:,:2]#subset_at_curr_inds(clean,inds,mask,num)
            # dpatches = denoise_subset(npatches,sigma)
            # psnr_log = log.create_psnr_log(dpatches,cpatches,10,3)
            # psnr_log = log.create_psnr_log(subset,clean,10,5)

            print(edges_log)
            print(l2vals_log)
            # print(psnr_log)


        #
        # -- update the indices in the subset --
        #

        if cnum <= max_ave:

            # -- set inds[:,0] = -1  if we don't want to grow --
            # print(inds[:,0])
            filter_inds(inds,data,mask,ave,sigma,quants,max_ave,cnum,thresh,pshape)
            # print(inds[:,0])

            # -- update mask --
            valid = th.nonzero(inds[:,0]!=-1)[:,0]
            print("valid.shape: ",valid.shape)
            sel_inds = th.gather(db_inds,1,inds[:,[0]])
            order[valid,cnum] = sel_inds[valid,0]
            # update_mask(mask[valid],inds[valid,[0]],valid)
            update_mask(mask,inds[:,[0]])

            # -- update total --
            num_add = 1
            cnum += num_add
        else:
            # -- update mask --
            nfill = snum - int(mask[0].sum().item())
            if nfill == 0: break
            update_mask(mask,inds[:,:nfill])

            sel_inds = th.gather(db_inds,1,inds[:,:nfill])
            order[:,cnum:cnum+nfill] = sel_inds

            # -- update total --
            num_add = nfill
            cnum += num_add

    # -- finalize --
    # subset = subset_data(data,mask == 1) # in the subset
    subset = subset_data(data,order,'inc') # in the subset
    vals,_ = l2_search_masked(data,mask,subset[:,[0]],sigma)
    # vals,_ = l2_search(subset,subset[:,[0]],sigma)
    # inds = th.nonzero(mask == 1)[:,1]
    # inds = rearrange(inds,'(b n) -> b n',b=bsize)
    inds = order[:,:snum]
    # csubset = subset_data(clean,mask == 1) # [clean data] in the subset
    csubset = subset_data(clean,order,'inc') # [clean data] in the subset
    # print("subset.shape: ",subset.shape)

    # -- inds needs to be ordered because index 0 must be the correct index. --

    return vals,inds,subset,csubset


def denoise_subset(noisy,sigma):

    # -- expand patches --
    patches = expand_patches(noisy)

    # -- exec denoising --
    patches_p = patches.clone()
    denoise_patches(patches,sigma)
    deno = patches

    return deno
