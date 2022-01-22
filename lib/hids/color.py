"""
RGB <-> YUV

"""

# -- linalg --
import numpy as np
import torch as th

# -- project --
from .utils import clone

def yuv2rgb(images):

    # -- shape --
    b,c,h,w = images.shape

    # -- weights --
    w = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]

    # -- copy channels --
    y = clone(images[:,0])
    u = clone(images[:,1])
    v = clone(images[:,2])

    # -- yuv -> rgb --
    images[:,0,...] = w[0] * y + w[1] * u + w[2] * 0.5 * v
    images[:,1,...] = w[0] * y - w[2] * v
    images[:,2,...] = w[0] * y - w[1] * u + w[2] * 0.5 * v

def rgb2yuv(images):

    # -- shape --
    b,c,h,w = images.shape

    # -- weights --
    weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]

    # -- copy channels --
    r = clone(images[:,0])
    g = clone(images[:,1])
    b = clone(images[:,2])

    # -- rgb -> yuv --
    images[:,0] = weights[0] * (r + g + b)
    images[:,1] = weights[1] * (r - b)
    images[:,2] = weights[2] * (.25 * r - .5 * g + .25 * b)

