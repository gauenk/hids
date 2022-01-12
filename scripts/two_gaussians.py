
# -- python --
import torch as th
from itertools import chain

# -- project --
import hids
from hids import testing

def two_gaussians():

    # -- create experiment fields --
    fields = {"mean":[0],"mean_cat":["lb-ub"],"mbounds":[(0.01,0.1)],
              "cov_cat":["diag","strong_tri","weak_tri"],"hypoType":["hot","mmd"],
              "sigma":[0.2,0.1,0.05],"bsize":[128],"num":[500],"dim":[98],"snum":[100]}
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
        cg_vals,cg_inds = hids.subset_search(noisy,sigma,snum,"coreset")

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_search(noisy,sigma,snum,"rand-hypo",
                                             hypoType=hypoType)

        # -- gradient based --
        gh_vals,gh_inds = hids.subset_search(noisy,sigma,snum,"grad-hypo",
                                             hypoType=hypoType)

        # -- compare --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds)
        cg_cmp = hids.compare_inds(gt_inds,cg_inds)
        rh_cmp = hids.compare_inds(gt_inds,rh_inds)
        gh_cmp = hids.compare_inds(gt_inds,gh_inds)
        print(l2_cmp)
        print(cg_cmp)

        # -- get results --
        inds = {'gt_inds':gt_inds,'l2_inds':l2_inds,'cg_inds':cg_inds,
                'rh_inds':rh_inds,'gh_inds':gh_inds,}
        cmps = {'l2_cmp':l2_cmp,'cg_inds':cg_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps)))
        results.append(result)
        # print(results)

    return results

if __name__ == "__main__":
    two_gaussians()
