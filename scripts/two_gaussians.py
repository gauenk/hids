
# -- python --
import tqdm
from itertools import chain

# -- data management --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- project --
import hids
from hids import testing

def two_gaussians():

    # -- create experiment fields --
    device = 'cuda:0'
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
        snum = exp['snum']
        sigma = exp["sigma"]
        means_eq = exp['means_eq']
        seed = exp['seed']

        # -- set random seed --
        np.random.seed(seed)
        th.manual_seed(seed)

        # -- get data --
        noisy = th.load("../vnlb/data/patches_noisy.pt")/255.
        clean = th.load("../vnlb/data/patches_clean.pt")/255.
        B,N = noisy.shape[:2]
        noisy = noisy.reshape(B,N,-1)*2
        clean = clean.reshape(B,N,-1)*2
        # noisy,clean = testing.load_gaussian_data(*ds_args)
        noisy,clean = noisy.to(device),clean.to(device)

        # -- info on clean data --
        delta_m = th.mean((clean[:,[0]] - clean)**2,2).mean(1)
        delta_s = th.mean((clean[:,[0]] - clean)**2,2).std(1)
        print(delta_m,delta_s)

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

        # -- coreset growth --
        gt_order = hids.subset_search(clean,0.,num,"l2")[1]
        cg_vals,cg_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search = 2*means_eq,
                                             max_mindex = means_eq)
        #, l2_order=gt_order)

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search = means_eq,
                                             max_mindex=means_eq)#means_eq)

        # -- gradient based --
        # gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"grad-hypo",
        #                                      hypoType=hypoType)
        gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search=2)
        # -- too many iters! --
        tm_vals,tm_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search=5,max_mindex=means_eq)

        # -- compare --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds)
        cg_cmp = hids.compare_inds(gt_inds,cg_inds)
        rh_cmp = hids.compare_inds(gt_inds,rh_inds)
        gh_cmp = hids.compare_inds(gt_inds,gh_inds)
        tm_cmp = hids.compare_inds(gt_inds,tm_inds)
        # print("l2: ",l2_cmp)
        # print("rh: ",rh_cmp)
        # print("cg: ",cg_cmp)

        # -- get results --
        # inds = {'gt_inds':gt_inds,'l2_inds':l2_inds,'cg_inds':cg_inds,
        #         'rh_inds':rh_inds,'gh_inds':gh_inds,}
        # cmps = {'l2_cmp':l2_cmp,'cg_inds':cg_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}

        # -- abbr. results --
        inds = {}
        cmps = {'l2_cmp':l2_cmp,'cg_cmp':cg_cmp,'rh_cmp':rh_cmp,
                'gh_cmp':gh_cmp,'tm_cmp':tm_cmp}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps)))
        results.append(result)
        # print(results)

    # -- format and print --
    results = pd.DataFrame(results)
    # pkeys = ['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp',
    #          'tm_cmp','mbounds','means_eq','seed']
    pkeys = ['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp','tm_cmp']
    print(results[pkeys])

    return results

if __name__ == "__main__":
    two_gaussians()
