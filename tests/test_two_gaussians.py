
# -- python --
import cv2,tqdm,copy
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch,time
from einops import rearrange,repeat

# -- package --
import hids

# -- testing --
import unittest

#
# -- Primary Testing Class --
#

class TestTwoGaussians(unittest.TestCase):

    def load_gaussian_data(self,mean,cov_cat,sigma,bsize,num,dim):
        """
        X ~ N(A,B)
        """

        # -- get covariance
        samples = hids.testings.sample_cov_cat(cov_cat,sigma,bsize,num,dim)

        # -- add mean --
        samples = samples + mean

        return samples

    def load_noniid_gaussian_data(self,mean_lb,mean_up,cov_cat,sigma,bsize,num,dim):

        # -- get covariance
        samples = hids.testing.sample_cov_cat(cov_cat,sigma,bsize,num,dim)

        # -- sample means --
        means = hids.testing.sample_means(mean_lb,mean_ub)

        # -- add mean --
        samples = samples + means

        return samples

    def test_two_gaussians(self):

        # -- get data --
        mean = 0
        cov_cat = "diag"
        sigma = 0.1
        bsize = 128
        num = 500
        dim = 98
        snum = 100
        data = self.load_gaussian_data(mean,cov_cat,sigma,bisze,num,dim)

        # -- l2 --
        vals,inds = hids.subset_data(data,sigma,snum,"l2")

        # -- hids-mmd --
        vals,inds = hids.subset_data(data,sigma,snum,"hids-mmd")

    def test_two_noniid_gaussians(self):

        # -- get data --
        mlb,mup = 0.01,10
        cov_cat = "diag"
        snum = 100
        sigma = 0.1
        bsize,num,dim = 128,500,98
        data = self.load_noniid_gaussian_data(mlb,mup,cov_cat,sigma,bsize,num,dim)

        # -- l2 --
        vals,inds = hids.subset_data(data,sigma,snum,"l2")

        # -- hids-mmd --
        vals,inds = hids.subset_data(data,sigma,snum,"hids-mmd")

