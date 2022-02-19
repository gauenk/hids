"""

Run comparison tests on image patches

"""


# -- python --
import tqdm,math
from pathlib import Path
from itertools import chain

# -- load the data --
from datasets import load_dataset

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
from vpss import get_patches

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

def img_patches():

    # -- create experiment fields --
    device = 'cuda:0'
    fields = {"sigma":[10.,30.,50.],"dataset":["davis"],
              "ps": [11],"npatches":[10],"nneigh":[15],
              "snum":[100],"num":[500],"seed":[123]
    }
    exps = testing.get_exp_mesh(fields)

    # -- init results --
    results = []
    for exp in tqdm.tqdm(exps):

        # -- unpack --
        num = exp['num']
        snum = exp['snum']
        sigma = exp["sigma"]
        seed = exp['seed']
        ps = exp['ps']
        nneigh = exp['nneigh']
        npatches = exp['npatches']

        # -- init config --
        exp.batch_size = 4
        exp.frame_size = [128,128]
        exp.noise_level = exp.sigma

        # -- set random seed --
        np.random.seed(seed)
        th.manual_seed(seed)

        # -- load dataset --
        data,loaders = load_dataset(exp)

        # -- get sample pair --
        sample = next(iter(loaders.tr))
        noisy = sample['dyn_noisy']
        clean = sample['dyn_clean']
        noisy = noisy.to(device)
        clean = clean.to(device)

        # -- get vnlb basic image --
        # basic = vnlb.get_basic_est(noisy,sigma)
        basic = noisy.clone()

        # -- sample patches --
        pt = 1
        pnoisy,pclean,sample_inds = hids.imgs2patches(noisy,clean,sigma,ps,
                                                      npatches,num,**{'pt':pt})

        # -- get remaining patches --
        pbasic = get_patches(basic,sample_inds,ps,mode="batch",pt=pt)
        pbasic_7 = get_patches(basic,sample_inds,3,mode="batch",pt=pt)
        pnoisy_7 = get_patches(noisy,sample_inds,3,mode="batch",pt=pt)
        pclean_7 = get_patches(clean,sample_inds,3,mode="batch",pt=pt)
        nps = 25
        pnoisy_nd = get_patches(noisy,sample_inds,nps,mode="batch",pt=pt)
        pclean_nd = get_patches(clean,sample_inds,nps,mode="batch",pt=pt)

        # -- correct shape --
        pnoisy = rearrange(pnoisy,'b p n t c h w -> (b p) n (t c h w)')
        pbasic = rearrange(pbasic,'b p n t c h w -> (b p) n (t c h w)')
        pclean = rearrange(pclean,'b p n t c h w -> (b p) n (t c h w)')
        pnoisy_7 = rearrange(pnoisy_7,'b p n t c h w -> (b p) n (t c h w)')
        pbasic_7 = rearrange(pbasic_7,'b p n t c h w -> (b p) n (t c h w)')
        pclean_7 = rearrange(pclean_7,'b p n t c h w -> (b p) n (t c h w)')
        pnoisy_nd = rearrange(pnoisy_nd,'b p n t c h w -> (b p) n (t c h w)')
        pclean_nd = rearrange(pclean_nd,'b p n t c h w -> (b p) n (t c h w)')
        print("pnoisy.shape: ",pnoisy.shape)
        print("pbasic.shape: ",pbasic.shape)
        print("pclean.shape: ",pclean.shape)
        print("pnoisy_7.shape: ",pnoisy_7.shape)
        print("pbasic_7.shape: ",pbasic_7.shape)
        print("pclean_7.shape: ",pclean_7.shape)

        # -- reorder samples according to l2 --
        reorder_bool = False
        if reorder_bool:
            rnum = pclean_7.shape[-2]
            reorder_inds = hids.subset_search(pclean_7*255.,0.,rnum,"l2")[1]

            D = pnoisy.shape[2]
            aug_inds = repeat(reorder_inds,'b n -> b n d',d=D)
            pnoisy = th.gather(pnoisy,1,aug_inds)
            pbasic = th.gather(pbasic,1,aug_inds)
            pclean = th.gather(pclean,1,aug_inds)

            D7 = pnoisy_7.shape[2]
            aug_inds = repeat(reorder_inds,'b n -> b n d',d=D7)
            pnoisy_7 = th.gather(pnoisy_7,1,aug_inds)
            pbasic_7 = th.gather(pbasic_7,1,aug_inds)
            pclean_7 = th.gather(pclean_7,1,aug_inds)

            print("-="*35)
            print("pnoisy.shape: ",pnoisy.shape)
            print("pbasic.shape: ",pbasic.shape)
            print("pclean.shape: ",pclean.shape)
            print("pnoisy_7.shape: ",pnoisy_7.shape)
            print("pbasic_7.shape: ",pbasic_7.shape)
            print("pclean_7.shape: ",pclean_7.shape)
            print("-="*35)

        # -- rescale --
        sigma = sigma/255.

        # -- gt --
        gt_vals,gt_inds = hids.subset_search(pclean_7*255.,0.,snum,"l2")

        # -- l2 --
        l2_vals,l2_inds = hids.subset_search(pnoisy,sigma,snum,"l2")

        # -- l2 --
        l27_vals,l27_inds = hids.subset_search(pnoisy_7,sigma,snum,"l2")

        # -- needle --
        nd_vals,nd_inds = hids.subset_search(pnoisy_nd,sigma,snum,"needle",ps=nps)
        # nd_vals,nd_inds = hids.subset_search(pclean_nd,sigma,snum,"needle",ps=nps)

        # -- basic --
        vb_vals,vb_inds = hids.subset_search(pbasic,sigma,snum,"l2")

        # -- basic --
        vb7_vals,vb7_inds = hids.subset_search(pbasic_7,sigma,snum,"l2")

        # -- ours [v1] --
        # v1_vals,v1_inds = hids.subset_search(pnoisy,sigma,snum,"beam",
        #                                      bwidth=10,swidth=10,
        #                                      num_search = 10, max_mindex=3,
        #                                      svf_method="svar")
        v1_inds = l2_inds.clone()

        # -- ours [v2] --
        # v2_vals,v2_inds = hids.subset_search(pnoisy,sigma,snum,"beam",
        #                                      bwidth=10,swidth=10,
        #                                      num_search=10, max_mindex=3)
        v2_inds = l2_inds.clone()

        # -- compare inds --
        l2_cmp = hids.compare_inds(gt_inds,l2_inds,False)
        l27_cmp = hids.compare_inds(gt_inds,l27_inds,False)
        nd_cmp = hids.compare_inds(gt_inds,nd_inds,False)
        vb_cmp = hids.compare_inds(gt_inds,vb_inds,False)
        vb7_cmp = hids.compare_inds(gt_inds,vb7_inds,False)
        v1_cmp = hids.compare_inds(gt_inds,v1_inds,False)
        v2_cmp = hids.compare_inds(gt_inds,v2_inds,False)

        # print("-="*25)
        # print(gt_inds[4])
        # print(th.sort(l2_inds[4]).values)
        # print(th.sort(l2_inds[4]).values)
        # print("-="*25)

        print("-"*30)
        print("-"*30)
        print(l2_cmp,l2_cmp.mean())
        print(l27_cmp,l27_cmp.mean())
        print(nd_cmp,nd_cmp.mean())
        print(vb_cmp,vb_cmp.mean())
        print(vb7_cmp,vb7_cmp.mean())
        print(v1_cmp,v1_cmp.mean())
        print(v2_cmp,v2_cmp.mean())

        # print("-="*20)
        # print(th.stack([l2_cmp,v1_cmp],-1))
        # print(th.stack([l2_cmp,nd_cmp],-1))
        # print("-="*20)

        # -- compare psnr --
        gt_psnr = hids.psnr_at_inds(noisy,clean,gt_inds)
        l2_psnr = hids.psnr_at_inds(noisy,clean,l2_inds)
        nd_psnr = hids.psnr_at_inds(noisy,clean,nd_inds)
        l27_psnr = hids.psnr_at_inds(noisy,clean,l27_inds)
        v1_psnr = hids.psnr_at_inds(noisy,clean,v1_inds)
        v2_psnr = hids.psnr_at_inds(noisy,clean,v2_inds)

        # -- get results --
        # inds = {'gt_inds':gt_inds,'l2_inds':l2_inds,'cg_inds':cg_inds,
        #         'rh_inds':rh_inds,'gh_inds':gh_inds,}
        # cmps = {'l2_cmp':l2_cmp,'cg_inds':cg_cmp,'rh_cmp':rh_cmp,"gh_cmp":gh_cmp}

        # -- abbr. results --
        inds = {}
        cmps = {'l2_cmp':l2_cmp,'l27_cmp':l27_cmp,'nd_cmp':nd_cmp,
                'v1_cmp':v1_cmp,'v2_cmp':v2_cmp}
        cmps = {k:v.mean().item() for k,v in cmps.items()}
        psnr = {'gt_psnr':gt_psnr,'l2_psnr':l2_psnr,'l27_psnr':l27_psnr,
                'nd_psnr':nd_psnr,'v1_psnr':v1_psnr,'v2_psnr':v2_psnr}
        result = dict(chain.from_iterable(d.items() for d in (exp,inds,cmps,psnr)))
        results.append(result)
        # print(results)

    # -- format and print --
    results = pd.DataFrame(results)
    # pkeys = ['sigma','l2_cmp','cg_cmp','rh_cmp','gh_cmp',
    #          'tm_cmp','mbounds','means_eq','seed']
    pkeys = ['sigma','l2_cmp','l27_cmp','nd_cmp','v1_cmp','v2_cmp']
    print(results[pkeys])
    # pkeys = ['sigma','gt_psnr','l2_psnr','nd_psnr','v1_psnr','v2_psnr']
    # print(results[pkeys])

    return results

if __name__ == "__main__":
    img_patches()

