
import torch as th
import numpy as np

# -- local --
from .sample_cov import sample_cov_cat
from .sample_mean import sample_mean_cat

def load_gaussian_data(mean_cat,mean,mbounds,cov_cat,sigma,bsize,num,dim):

    # -- cov samples --
    cov_samples = sample_cov_cat(cov_cat,sigma,bsize,num,dim)

    # -- mean samples --
    mean_samples = sample_mean_cat(mean_cat,mean,mbounds,bsize,num,dim)

    # -- create samples --
    noisy = cov_samples + mean_samples
    clean = mean_samples

    return noisy,clean

