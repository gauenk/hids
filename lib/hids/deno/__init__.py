
# -- vnlb --
from vnlb.gpu.bayes_est import bayes_estimate_batch

# -- local --
from .bayes import bayes_deno
from .wnnm import wnnm_deno

def denoise_subset(patches,sigma,method="wnnm"):
    patches = patches.clone()
    deno = denoise_patches(patches,sigma,method="wnnm")
    return deno

def denoise_patches(patches,sigma,method="wnnm"):
    if method == "bayes":
        return bayes_deno(patches,sigma)
    elif method == "wnnm":
        return wnnm_deno(patches,sigma)
    else:
        raise ValueError(f"Uknown denoiser method [{method}]")
