
import hids

def two_gaussians(mean,cov_cat,sigma,bisze,num,dim,snum):

    # -- create experiment fields --
    fields = {"mean":[0],"mean_cat":["ave","lb-ub"],"mbounds":[(0.01,10)],
              "cov_cat":["diag","strong_tri","weak_tri"],"hypoType":["hot","mmd"],
              "sigma":[0.2,0.1,0.05],"bsize":[128],"num":[500],"dim":[98],"snum":[100]}
    exps = hids.testing.get_exp_mesh(fields)
    ds_fields = ["mean","cov_cat","sigma","bsize","num","dim"]

    # -- init results --
    results = []
    for exp in exps:

        # -- unpack --
        ds_args = [exp[field] for field in ds_fields]
        hypoType = exp['hypoType']
        snum = exp['snum']

        # -- get data --
        clean,noisy = hids.testing.load_gaussian_data(*ds_args)

        # -- gt --
        gt_vals,gt_inds = hids.subset_data(clean,sigma,snum,"l2")

        # -- l2 --
        l2_vals,l2_inds = hids.subset_data(noisy,sigma,snum,"l2")

        # -- random subsetting --
        rh_vals,rh_inds = hids.subset_data(noisy,sigma,snum,"rand-hypo",
                                           hypoType=hypoType)

        # -- gradient based --
        gh_vals,gh_inds = hids.subset_data(noisy,sigma,snum,"grad-hypo",
                                           hypoType=hypoType)

        # -- compare --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds)
        rh_cmp = hids.compare_inds(gt_inds,rh_inds)
        gh_cmp = hids.compare_inds(gt_inds,gh_inds)

        # -- get results --
        inds = {'l2_inds':l2_inds,'rh_inds':rh_inds,'gh_inds':gh_inds,'gt_inds':gt_inds}
        cmps = {'l2_cmp':l2_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}
        result = exp + inds + cmps
        results.append(result)
    return results
