
# -- linalg --
from einops import rearrange,repeat

# -- patch-based neural network --
# import pdnn

# -- package --
from hids.patch_utils import yuv2rgb_patches,rgb2yuv_patches

# -- local --
from .bayes import bayes_deno
from .wnnm import wnnm_deno

def denoise_subset(patches,sigma,method="wnnm"):
    patches = patches.clone()
    deno = denoise_patches(patches,sigma,method)
    return deno

def denoise_patches(patches,sigma,method="wnnm",**kwargs):
    if method == "bayes":
        return bayes_deno(patches,sigma)
    elif method == "wnnm":
        return wnnm_deno(patches,sigma)
    elif method == "pdnn":
        return deno_pdnn(patches,sigma,**kwargs)
    else:
        raise ValueError(f"Uknown denoiser method [{method}]")


def deno_pdnn(patches,sigma,ps=7):

    # -- yuv 2 rgb --
    rgb_patches = yuv2rgb_patches(patches)

    # -- denoise --
    # rgb_deno = pdnn.denoise_patches(rgb_patches,sigma,13)
    rgb_deno = rgb_patches

    # -- crop to actual search ps --
    assert 13 >= ps
    psh = (13 - ps)//2
    rgb_deno = rgb_deno[...,psh:-psh,psh:-psh]

    # -- rgb 2 yuv --
    t,c,h,w = rgb_deno.shape[2:]
    rgb_deno = rearrange(rgb_deno,'b n t c h w -> b n (t c h w)')
    deno = rgb2yuv_patches(rgb_deno)
    deno = rearrange(deno,'b n (t c h w) -> b n t c h w',t=t,c=c,h=h)

    return deno
