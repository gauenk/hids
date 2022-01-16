

# -- imports --
import torch,math
import torch as th
from torch import nn
from torch import optim
import numpy as np
torch.autograd.set_detect_anomaly(True)
from einops import rearrange,repeat


class WeightNet(nn.Module):

    def __init__(self,bsize,nsamples,dim):
        super(WeightNet, self).__init__()

        iweights = torch.ones(bsize,nsamples,1)
        # iweights = torch.zeros(bsize,nsamples,1)
        # iweights = 3*torch.ones(bsize,nsamples,1)
        # iweights[:,:30] = 1.

        self.relu = nn.ReLU()
        self.weights = nn.Parameter(iweights)
        # self.weights = nn.Parameter(torch.rand(bsize,nsamples,1)-0.5)
        self.bias = nn.Parameter(torch.zeros(bsize,dim))
        # self.bias = nn.Parameter(torch.zeros(bsize,nsamples))

    def forward(self,x):

        # -- weight samples --
        weights = self.weights
        expw = torch.exp(-weights)
        weighted_x = expw * x

        # -- compute average across weights  --
        ave_wx = torch.mean(weighted_x,dim=1)

        return weighted_x,ave_wx

    def freeze_bias(self):
        self.bias.requires_grad = False

    def defrost_bias(self):
        self.bias.requires_grad = True

    def freeze_weights(self):
        self.weights.requires_grad = False

    def defrost_weights(self):
        self.weights.requires_grad = True

    def get_weights(self):
        return self.weights,self.bias

    def clamp_weights(self):
        self.weights.data.clamp_(0)

    def round_weights(self,nkeep=30):
        vals,inds = torch.topk(-self.weights.data[...,0],nkeep,1)
        self.weights.data[...] = 0.
        self.weights.data[...,0].scatter_(1,inds,1.)

    def get_topk_patches(self,nkeep=30):
        vals,inds = torch.topk(-self.weights.data[...,0],nkeep,1)
        return inds
