
# -- python --
import numpy as np
import torch as th
import pandas as pd
from itertools import chain

# -- project --
import hids
from hids import testing

def two_gaussians():

    # -- create experiment fields --
    fields = {"mean":[0],"mean_cat":["lb-ub"],"mbounds":[(0.01,1.)],
              "cov_cat":["diag"],"hypoType":["hot"],
              "sigma":[0.2,0.05],"bsize":[128],"num":[500],"dim":[98],"snum":[100]}
    exps = testing.get_exp_mesh(fields)
    ds_fields = ["mean_cat","mean","mbounds","cov_cat","sigma","bsize","num","dim"]

    # -- init results --
    results = []
    for exp in exps:

        # -- unpack --
        ds_args = [exp[field] for field in ds_fields]
        hypoType = exp['hypoType']
        snum = exp['snum']
        sigma = exp["sigma"]

        # -- get data --
        noisy,clean = testing.load_gaussian_data(*ds_args)

        # -- gt --
        gt_vals,gt_inds = hids.subset_search(clean,0.,snum,"l2")

        # -- l2 --
        l2_vals,l2_inds = hids.subset_search(noisy,sigma,snum,"l2")

        # -- coreset growth --
        cg_vals,cg_inds = hids.subset_search(noisy,sigma,snum,"l2")

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_search(noisy,sigma,snum,"beam",
                                             hypoType=hypoType)

        print("rh_inds.shape: ",rh_inds.shape)
        print("l2_inds.shape: ",l2_inds.shape)

        # -- gradient based --
        # gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"grad-hypo",
        #                                      hypoType=hypoType)
        gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"l2",
                                             hypoType=hypoType)

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
        cmps = {'l2_cmp':l2_cmp,'rh_cmp':rh_cmp}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps)))
        results.append(result)
        # print(results)
    print(results)
    results = pd.DataFrame(results)
    print(results[['sigma','l2_cmp','rh_cmp','mbounds']])

    return results

if __name__ == "__main__":
    np.random.seed(123)
    th.manual_seed(123)
    two_gaussians()
