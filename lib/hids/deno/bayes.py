

# -- vnlb --
from vnlb.gpu.bayes_est import bayes_estimate_batch
from .utils import expand_patches


def bayes_deno(patches,sigma):

    patches = expand_patches(patches)
    group_chnls,cs,cs_ptr = 1,-1,-1
    sigma2 = sigma**2
    sigmab2 = sigma**2
    rank = 39
    thresh = .0001
    # thresh = 2.7
    step = 0
    flat_patch = th.zeros(patches.shape[0],device=patches.device)
    patches_basic = th.zeros_like(patches)
    patches_clean = None#th.zeros_like(patches)
    bayes_estimate_batch(patches,patches_basic,patches_clean,sigma2,
                         sigmab2,rank,False,thresh,step==1,flat_patch,cs,cs_ptr)





