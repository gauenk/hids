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

def sample_var(vals,data,params):
    sigma = optional(params,'sigma',0.)
    mean = data.mean(-2,keepdim=True)
    vals[...] = ((data - mean)**2).mean((-2,-1))
    vals[...] = th.abs(vals - sigma)
    return vals
