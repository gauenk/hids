
from .l2_search import l2_search,l2_subset
from .coreset_growth import exec_coreset_growth,exec_beam_search
from .rand_hypo import rand_hypo
from .grad_hypo import grad_hypo
from .utils import gather_data,compare_inds


def subset_search(data,sigma,snum,method,**kwargs):
    if method == "l2":
        return l2_subset(data,sigma,snum)
    elif method == "coreset":
        return exec_coreset_growth(data,sigma,snum,**kwargs)
    elif method == "rand-hypo":
        return rand_hypo(data,sigma,snum,**kwargs)
    elif method == "grad-hypo":
        return grad_hypo(data,sigma,snum,**kwargs)
    elif method == "beam":
        return exec_beam_search(data,sigma,snum,**kwargs)
    else:
        raise ValueError(f"Uknown method [{method}]")
