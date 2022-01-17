
# -- python --
import numpy as np
import torch as th
import pandas as pd
from itertools import chain
from einops import rearrange,repeat

# -- project --
import hids
from hids import testing

def two_gaussians():

    # -- create experiment fields --
    fields = {"mean":[0],"mean_cat":["lb-ub"],
              #"mbounds":[(0.01,5.),(0.01,.1),(0.01,.5),(0.01,1.)],
              "mbounds":[(0.01,.5)],
              "means_eq":[5,10],"cov_cat":["diag"],"hypoType":["hot"],
              "sigma":[0.2],"bsize":[128],"num":[500],"dim":[98*3],"snum":[100]}
    exps = testing.get_exp_mesh(fields)
    ds_fields = ["mean_cat","mean","mbounds","cov_cat","sigma",
                 "bsize","num","dim","means_eq"]

    # -- init results --
    results = []
    for exp in exps:

        # -- unpack --
        ds_args = [exp[field] for field in ds_fields]
        hypoType = exp['hypoType']
        num = exp['num']
        snum = exp['snum']
        sigma = exp["sigma"]
        means_eq = exp['means_eq']

        # -- get data --
        noisy,clean = testing.load_gaussian_data(*ds_args)

        # -- gt --
        gt_vals,gt_inds = hids.subset_search(clean,0.,snum,"l2")

        # -- l2 --
        l2_vals,l2_inds = hids.subset_search(noisy,sigma,snum,"l2")

        # -- coreset growth --
        gt_order = hids.subset_search(clean,0.,num,"l2")[1]
        cg_vals,cg_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search = 10, l2_order=gt_order)

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search = means_eq)

        # -- gradient based --
        # gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"grad-hypo",
        #                                      hypoType=hypoType)
        gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             num_search=30)

        # -- compare --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds)
        cg_cmp = hids.compare_inds(gt_inds,cg_inds)
        rh_cmp = hids.compare_inds(gt_inds,rh_inds)
        gh_cmp = hids.compare_inds(gt_inds,gh_inds)
        # print("l2: ",l2_cmp)
        # print("rh: ",rh_cmp)
        # print("cg: ",cg_cmp)

        # -- get results --
        # inds = {'gt_inds':gt_inds,'l2_inds':l2_inds,'cg_inds':cg_inds,
        #         'rh_inds':rh_inds,'gh_inds':gh_inds,}
        # cmps = {'l2_cmp':l2_cmp,'cg_inds':cg_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}

        # -- abbr. results --
        inds = {}
        cmps = {'l2_cmp':l2_cmp,'cg_cmp':cg_cmp,'rh_cmp':rh_cmp,'gh_cmp':gh_cmp}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps)))
        results.append(result)
        # print(results)

    # -- format and print --
    results = pd.DataFrame(results)
    print(results[['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp','mbounds','means_eq']])

    return results

if __name__ == "__main__":
    np.random.seed(123)
    th.manual_seed(123)
    two_gaussians()
