# -- linalg --
import numpy as np
from einops import rearrange,repeat


def expand_patches(patches,c=3,pt=2):
    if patches.dim() == 3:
        shape_str = 'b n (pt c ph pw) -> b n pt c ph pw'
        b,n,dim = patches.shape
        ps = int(np.sqrt(dim // (c*pt)))
        patches = rearrange(patches,shape_str,c=c,pt=pt,ph=ps)
    return patches
