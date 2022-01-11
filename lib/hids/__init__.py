
from .l2_search import l2_search
from .rand_hypo import rand_hypo


def subset_data(data,sigma,snum,method,**kwargs):
    if method == "l2":
        return l2_search(data,sigma,snum)
    elif method == "rand-hypo":
        return rand_hypo(data,sigma,snum,**kwargs)
    elif method == "grad-hypo":
        return grad_hypo(data,sigma,snum,**kwargs)
    else:
        raise ValueError(f"Uknown method [{method}]")
