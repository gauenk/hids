"""
State value functions

"""

# -- linalg --
import torch as th
import numpy as np

# -- local --
from hids.utils import optional

def get_state_value_function(method):
    if method == "svar":
        return sample_var
    elif method == "svar_blur":
        return sample_var_blur
    else:
        raise ValueError(f"Update reference [{method}]")

def sample_var(data,params):
    sigma = optional(params,'sigma',0.)
    print("data.shape: ",data.shape)
    mean = data.mean(-2,keepdim=True)
    print("mean.shape: ",mean.shape)
    delta = (data - mean)**2
    print("delta.shape: ",delta.shape)
    delta = delta.mean((-2,-1))
    delta = th.abs(delta - sigma)
    print("delta.shape: ",delta.shape)
    return delta
