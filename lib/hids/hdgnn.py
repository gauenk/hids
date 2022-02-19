


# -- python --
import math

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange

# -- local --
from .utils import optional,compute_target_sigma
from .sobel import apply_sobel_to_patches

def exec_hdgnn(vals,data,cnum,params):

    # -- unpack --
    device = vals.device
    mindex = min(cnum,params.max_mindex)
    sigma = optional(params,'sigma',0.)
    t_sigma = compute_target_sigma(sigma,mindex)
    shape = list(data.shape)
    b,nb,ns,k,d = shape

    # -- estimate --
    mean,sigma = est_gaussian_params(data)

    # -- sim1 --
    N,B = 100,100
    Y = sim_gassian(mean,sigma,N)
    rYY = compute_nn(Y,data)
    # print("rYY.shape: ",rYY.shape)
    # print(rYY[0,0])
    # print(rYY[0,1])

    rYYs = th.zeros(B,b,nb,ns).to(device)
    for b in range(B):
        Xs = sim_gassian(mean,sigma,N)
        mean_s,sigma_s = est_gaussian_params(Xs)
        Ys = sim_gassian(mean_s,sigma_s,N)
        rYYs[b] = compute_nn(Ys,Xs)
    mrYYs = th.mean(rYYs,0,keepdim=True)

    # -- compare --
    dYYs = th.abs(mrYYs - rYYs)
    dYY = th.abs(mrYYs - rYY)
    # print(dYY[:3,0,0])
    # print(dYYs[:3,0,0])
    comp = (dYYs > dYY).type(th.float32)
    pvalue = th.mean(comp,0)
    # print("pvalue.shape: ",pvalue.shape)
    # print("vals.shape: ",vals.shape)
    # print(pvalue[0,0])

    # -- fill vals --
    vals[...] = pvalue

    return vals

def compute_nn(A,B):

    # -- unpack --
    device = A.device
    nA = A.shape[-2]
    nB = B.shape[-2]

    # -- data stack --
    S = th.cat([A,B],-2)
    T = S.shape[-2]

    # -- labels --
    lA = th.ones(A.shape[:-1])
    lB = th.zeros(B.shape[:-1])
    L = th.cat([lA,lB],-1).to(device)

    # -- compute pwd --
    D = th.cdist(S,S)

    # -- compute nn --
    order = th.sort(D,-1).indices[...,:nA,1]

    # -- compute perc --
    R = th.gather(L,-1,order)
    R = R.mean(-1)

    return R

def sim_gassian(mean,sigma,N):

    # -- unpack --
    device = mean.device
    shape = list(sigma.shape[:3])

    # -- contract --
    b,nb,ns,d1,d2 = sigma.shape
    sigma = rearrange(sigma,'b nb ns d1 d2 -> (b nb ns) d1 d2')

    # -- cholesky --
    R = th.linalg.cholesky(sigma)

    # -- N(0,1) --
    shape = (b*nb*ns,d1,N)
    Z = th.randn(size=shape).to(device)

    # -- Multi Gauss --
    X = th.matmul(R,Z)

    # -- expand --
    X = rearrange(X,'(b nb ns) d N -> b nb ns N d',b=b,nb=nb)

    # -- add mean --
    X += mean

    return X

def est_gaussian_params(data):
    b,nb,ns,k,d = data.shape
    mean = th.mean(data,-2,keepdim=True)
    left = rearrange(data - mean,'b nb ns k d -> (b nb ns) d k')
    right = rearrange(data - mean,'b nb ns k d -> (b nb ns) k d')
    sigma = th.matmul(left,right)/(k-1)
    alpha = 1e-5
    sigma[:,range(d),range(d)] += alpha
    sigma = rearrange(sigma,'(b nb ns) d1 d2 -> b nb ns d1 d2',b=b,nb=nb)
    return mean,sigma
