
# -- python --
import torch

# -- package --
from .utils import optional
from .l2_search import l2_subset

def rand_hypo(data,sigma,snum,**kwargs):
    return l2_subset(data,sigma,snum)
