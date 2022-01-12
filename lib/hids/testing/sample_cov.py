
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

def sample_cov_cat(cov_cat,sigma,bsize,num,dim):
    if cov_cat == "diag":
        return sample_cov_diag(sigma,bsize,num,dim)
    else:
        raise ValueError(f"Uknown covariance category [{cov_cat}]")

def sample_cov_diag(sigma,bsize,num,dim):
    samples = sigma * th.randn(bsize,num,dim)
    return samples
