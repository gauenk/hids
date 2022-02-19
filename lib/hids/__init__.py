
import torch as th
from .l2_search import l2_search,l2_subset
from .needle_search import needle_subset
from .coreset_growth import exec_coreset_growth
from .beam_search import exec_beam_search
from .rand_hypo import rand_hypo
from .grad_hypo import grad_hypo
from .utils import gather_data,compare_inds
# from .patching import imgs2patches
from .imgs2patches import imgs2patches

def subset_search(data,sigma,snum,method,**kwargs):
    if method == "l2":
        output = l2_subset(data,sigma,snum)
        th.cuda.empty_cache()
        return output
    elif method == "needle":
        output = needle_subset(data,sigma,snum)
        th.cuda.empty_cache()
        return output
    elif method == "coreset":
        return exec_coreset_growth(data,sigma,snum,**kwargs)
    elif method == "rand-hypo":
        return rand_hypo(data,sigma,snum,**kwargs)
    elif method == "grad-hypo":
        return grad_hypo(data,sigma,snum,**kwargs)
    elif method == "beam":
        output = exec_beam_search(data,sigma,snum,**kwargs)
        th.cuda.empty_cache()
        return output
    else:
        raise ValueError(f"Uknown method [{method}]")

def psnr_at_inds(noisy,clean,inds):

    return 0.
