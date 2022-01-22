
# -- python --
import tqdm,math
from pathlib import Path
from itertools import chain

# -- image --
import torchvision.utils as tv_utils

# -- data management --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- project --
import hids
from hids import testing

def save_patches(subset,name,shape):

    # -- filename --
    save_dir = Path("./output/")
    if not(save_dir.exists()):
        save_dir.mkdir()
    fn = str(save_dir / name)
    print("fn: ",fn)

    # -- reshape --
    t,c,h,w = shape
    subset = rearrange(subset,'n (t c h w) -> n t c h w',t=t,c=c,h=h)
    subset = subset[:,0]
    print("subset.shape: ",subset.shape)
    nrow = int(math.sqrt(subset.shape[0]))

    # -- save --
    tv_utils.save_image(subset.cpu(),fn,nrow=nrow,value_range=[0.,1.])

def two_gaussians():

    # -- create experiment fields --
    device = 'cuda:1'
    fields = {"mean":[0],"mean_cat":["lb-ub"],'seed':[123],
              #"mbounds":[(0.01,5.),(0.01,.1),(0.01,.5),(0.01,1.)],
              # "mbounds":[(0.01,.1),(0.01,.5),(0.01,1.)],
              "mbounds":[(0.01,.1)],
              "means_eq":[5],"cov_cat":["diag"],"hypoType":["hot"],
              "sigma":[30./255.],"bsize":[128],"num":[500],"dim":[98*3],"snum":[100]}
    exps = testing.get_exp_mesh(fields)
    ds_fields = ["mean_cat","mean","mbounds","cov_cat","sigma",
                 "bsize","num","dim","means_eq"]

    # -- init results --
    results = []
    for exp in tqdm.tqdm(exps):

        # -- unpack --
        ds_args = [exp[field] for field in ds_fields]
        hypoType = exp['hypoType']
        num = exp['num']
        snum = 10#exp['snum']
        sigma = exp["sigma"]
        means_eq = exp['means_eq']
        seed = exp['seed']

        # -- set random seed --
        np.random.seed(seed)
        th.manual_seed(seed)

        # -- get data --
        noisy = th.load("../vnlb/data/patches_noisy.pt")/255.
        clean = th.load("../vnlb/data/patches_clean.pt")/255.
        noisy = noisy[:50]
        clean = clean[:50]
        B,N = noisy.shape[:2]
        noisy = noisy.reshape(B,N,-1)
        clean = clean.reshape(B,N,-1)
        B,N,D = noisy.shape
        # noisy,clean = testing.load_gaussian_data(*ds_args)
        noisy,clean = noisy.to(device),clean.to(device)

        # -- info on clean data --
        delta_m = th.mean((clean[:,[0]] - clean)**2,2).mean(1)
        delta_s = th.mean((clean[:,[0]] - clean)**2,2).std(1)
        print(delta_m,delta_s)

        # -- reorder samples according to l2 --
        reorder_inds = hids.subset_search(noisy,sigma,num,"l2")[1]
        reorder_inds = repeat(reorder_inds,'b n -> b n d',d=D)
        noisy = th.gather(noisy,1,reorder_inds)
        clean = th.gather(clean,1,reorder_inds)

        # -- reorder --
        save_patches(noisy[-1],"noisy_09.png",(2,3,7,7))
        save_patches(clean[-1],"clean_09.png",(2,3,7,7))
        save_patches(noisy[1],"noisy_01.png",(2,3,7,7))
        save_patches(clean[1],"clean_01.png",(2,3,7,7))
        save_patches(noisy[0],"noisy_00.png",(2,3,7,7))
        save_patches(clean[0],"clean_00.png",(2,3,7,7))

        # -- gt --
        gt_vals,gt_inds = hids.subset_search(clean,0.,snum,"l2")

        # -- check std --
        check_std = True
        if check_std:
            B,N,D = noisy.shape
            aug_inds = repeat(gt_inds,'b n -> b n d',d=D)[:,:means_eq]
            data = th.gather(noisy,1,aug_inds)
            mean = data.mean(-2,keepdim=True)
            vals = (((data - mean)**2).mean((-2,-1)) * (N/(N-1.))).pow(0.5)
            mstd = vals.mean().item()
            est_sigma = hids.beam_search.compute_target_sigma(sigma,means_eq)
            # print(mstd,sigma,est_sigma)

        # -- l2 --
        l2_vals,l2_inds = hids.subset_search(noisy,sigma,snum,"l2")
        print(l2_inds[0])

        # -- coreset growth --
        gt_order = hids.subset_search(clean,0.,num,"l2")[1]
        # cg_vals,cg_inds = hids.subset_search(noisy,sigma,snum,"beam",
        #                                      num_search = 2*means_eq,
        #                                      max_mindex = means_eq)
        cg_inds = l2_inds.clone()

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search = 10,
                                             max_mindex=means_eq)#means_eq)

        # -- gradient based --
        # gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"grad-hypo",
        #                                      hypoType=hypoType)
        # gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"beam",
        #                                      num_search=2)
        gh_inds = l2_inds.clone()

        # -- too many iters! --
        tm_vals,tm_inds = hids.subset_search(noisy,sigma,snum,"beam",bwidth=10,
                                             num_search=20,max_mindex=means_eq)
        # tm_inds = l2_inds.clone()

        # -- compare inds --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds,False)
        cg_cmp = hids.compare_inds(gt_inds,cg_inds,False)
        rh_cmp = hids.compare_inds(gt_inds,rh_inds,False)
        gh_cmp = hids.compare_inds(gt_inds,gh_inds,False)
        tm_cmp = hids.compare_inds(gt_inds,tm_inds,False)
        print("-"*30)
        print("-"*30)
        print(l2_cmp,l2_cmp.mean(),l2_cmp.std())
        print(rh_cmp,rh_cmp.mean(),rh_cmp.std())
        print(tm_cmp,tm_cmp.mean(),tm_cmp.std())
        # print("l2: ",l2_cmp)
        # print("rh: ",rh_cmp)
        # print("cg: ",cg_cmp)

        # -- compare psnr --
        gt_psnr = hids.psnr_at_inds(noisy,clean,gt_inds)
        l2_psnr = hids.psnr_at_inds(noisy,clean,l2_inds)
        cg_psnr = hids.psnr_at_inds(noisy,clean,cg_inds)
        rh_psnr = hids.psnr_at_inds(noisy,clean,rh_inds)
        gh_psnr = hids.psnr_at_inds(noisy,clean,gh_inds)
        tm_psnr = hids.psnr_at_inds(noisy,clean,tm_inds)

        # -- get results --
        # inds = {'gt_inds':gt_inds,'l2_inds':l2_inds,'cg_inds':cg_inds,
        #         'rh_inds':rh_inds,'gh_inds':gh_inds,}
        # cmps = {'l2_cmp':l2_cmp,'cg_inds':cg_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}

        # -- abbr. results --
        inds = {}
        cmps = {'l2_cmp':l2_cmp,'cg_cmp':cg_cmp,'rh_cmp':rh_cmp,
                'gh_cmp':gh_cmp,'tm_cmp':tm_cmp}
        psnr = {'gt_psnr':gt_psnr,'l2_psnr':l2_psnr,'cg_psnr':cg_psnr,'rh_psnr':rh_psnr,
                 'gh_psnr':gh_psnr,'tm_psnr':tm_psnr}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps,psnr)))
        results.append(result)
        # print(results)

    # -- format and print --
    results = pd.DataFrame(results)
    # pkeys = ['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp',
    #          'tm_cmp','mbounds','means_eq','seed']
    pkeys = ['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp','tm_cmp']
    print(results[pkeys])
    # pkeys = ['sigma','gt_psnr','l2_psnr','cg_psnr','rh_psnr','gh_psnr','tm_psnr']
    # print(results[pkeys])

    return results

if __name__ == "__main__":
    two_gaussians()
