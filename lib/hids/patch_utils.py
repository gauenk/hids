
# -- python --
import math

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- vision --
from PIL import Image,ImageDraw,ImageFont
from torchvision import utils as tv_utils
from torchvision.transforms import Pad as tv_pad

# -- project --
from hids.utils import clone
from hids.color import yuv2rgb,rgb2yuv
# from hids.deno import denoise_patches,denoise_subset


#
# -- patch coloring --
#

def yuv2rgb_patches(patches,use_clone=False,c=3,pt=2):
    return recolor_patches(patches,yuv2rgb,use_clone,c,pt)

def rgb2yuv_patches(patches,use_clone=False,c=3,pt=2):
    return recolor_patches(patches,rgb2yuv,use_clone,c,pt)

def recolor_patches(patches,recolor,use_clone,c=2,pt=3):

    # -- shapes --
    if use_clone: patches = clone(patches)
    patches = contract_patches(patches)
    b,n,pdim = patches.shape
    ps = int(np.sqrt(pdim//(pt*c)))

    # -- reshape --
    shape_str = 'b n (pt c ph pw) -> (b n pt) c ph pw'
    patches = rearrange(patches,shape_str,c=c,pt=pt,ph=ps)

    # -- recolor --
    recolor(patches)

    # -- reshape --
    shape_str = '(b n pt) c ph pw -> b n (pt c ph pw)'
    patches = rearrange(patches,shape_str,b=b,n=n)

    return patches

#
# -- patch psnrs --
#

def patch_psnrs(patches_a,patches_b,imax=255.,to_rgb=False,prefix=""):

    # -- convert color --
    if to_rgb:
        patches_a = yuv2rgb_patches(patches_a,True)
        patches_b = yuv2rgb_patches(patches_b,True)
    print("patches_a.shape: ",patches_a.shape)
    print("patches_b.shape: ",patches_b.shape)

    # -- comp psnr --
    eps = 1e-8
    B,N = patches_a.shape[:2]
    delta = (patches_a/imax - patches_b/imax)**2
    delta = delta.reshape(B,N,-1).mean(-1).cpu().numpy()
    log_mse = np.ma.log10(1./(delta+eps)).filled(-np.infty)
    psnrs = 10 * log_mse

    # -- remove t dimension --
    # patches_a = remove_pt_dim(patches_a)
    # patches_b = remove_pt_dim(patches_b)
    name_a = "a" if len(prefix) == 0 else f"a-{prefix}"
    name_b = "b" if len(prefix) == 0 else f"b-{prefix}"
    save_patches(name_a,patches_a,psnrs,3,4)
    save_patches(name_b,patches_b,None,3,4)

    return psnrs

def contract_patches(patches,c=3,pt=2):
    if patches.dim() > 3:
        shape_str = 'b n pt c ph pw -> b n (pt c ph pw)'
        patches = rearrange(patches,shape_str)
    return patches

def expand_patches(patches,c=3,pt=2):
    if patches.dim() == 3:
        shape_str = 'b n (pt c ph pw) -> b n pt c ph pw'
        b,n,dim = patches.shape
        ps = int(np.sqrt(dim // (c*pt)))
        patches = rearrange(patches,shape_str,c=c,pt=pt,ph=ps)
    return patches


def save_patches(name,patches,psnrs,nB=3,nN=10,t=2,c=3):

    # -- to patches --
    b,n,d = patches.shape
    ps = int(np.sqrt(d//(t*c)))
    patches = rearrange(patches,'b n (t c h w) -> b n t c h w',t=t,c=c,h=ps)
    for bidx in range(nB):

        # -- file --
        fn = f"output/save_patches_{name}_{bidx}.png"
        spatches = patches[bidx,:nN,0]

        if psnrs is None:
            # -- save --
            # print(spatches.min(),spatches.max())
            tv_utils.save_image(spatches/255.,fn)#,value_range=[0.,255.])
        else:
            # -- save with psnr --
            spsnr = "%2.2f" % psnrs[bidx,0]
            image = tv_utils.make_grid(spatches,value_range=[0,255.],padding=3)
            image = tv_pad((0,10,0,0))(image)
            image = image.cpu().numpy()
            image = rearrange(image,'c h w -> h w c')
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 1)
            d1 = ImageDraw.Draw(image)
            d1.text((0,0), spsnr, fnt=fnt, fill=(255, 0, 0))
            image.save(fn)


def remove_pt_dim(patches,t=2,c=3):
    b,n,d = patches.shape
    ps = d//(t*c)
    patches = rearrange(patches,'b n (t c h w) -> b n t c h w',t=t,c=c,h=ps)
    patches = patches[:,:,0]
    patches = rearrange(patches,'b n c h w -> b n (c h w)')
    return patches

