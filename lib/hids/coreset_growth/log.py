"""

Create reports


"""

# -- linalg --
import torch as th
import numpy as np

# -- project --
from hids.l2_search import l2_search
from hids.sobel import apply_sobel_to_patches
from hids.patch_utils import patch_psnrs

def create_params_log(**kwargs):
    # -- info --
    log = "-"*30 + "\n"
    log += "max_ave: %d\n" % max_ave
    log += "thresh: %2.3e\n" % thresh
    log += "step: %d\n" % step
    log += "sigma: %2.2f\n" % sigma
    return log

def create_edges_log(subset,ave,sigma,pshape,nB,nN):

    # -- comp --
    s_edges = apply_sobel_to_patches(subset,pshape)
    ave_edges = apply_sobel_to_patches(ave,pshape)
    emean = th.mean(s_edges,1)[:nB].ravel()
    estd = th.std(s_edges,1)[:nB].ravel()
    elem = ave_edges[:nB].ravel()

    # -- message --
    log = "-- edges -- \n"
    log += str(s_edges[:nB,:nN]) + "\n"
    # log += str(emean) + "\n"
    # log += str(estd) + "\n"
    # log += str(elem) + "\n"

    return log

def create_l2vals_log(aset,ave,sigma,nB,nN):

    # -- comp --
    s_vals,_ = l2_search(aset,ave,sigma)#sigma)
    s_means = (s_vals).mean(1)[:nB].ravel()
    s_stds = (s_vals).std(1)[:nB].ravel()

    # -- message --
    log = "-- vals -- \n"
    log += str(s_vals[:nB,:nN]) + "\n"
    # log += str(s_means) + "\n"
    # log += str(s_stds) + "\n"
    # log += str(s_means/s_stds) + "\n"
    # for i in range(10):
    #     elem = th.median(s_vals[i]).ravel().item()
    #     log += str(elem) + "\n"
    return log

def create_psnr_log(subset,clean,nB,nN):

    # -- init message --
    log = "-- psnrs --\n"
    if clean is None:
        log += "clean is None\n"
        return log

    # -- comp --
    psnrs_0 = patch_psnrs(subset,clean,to_rgb=True)
    # psnrs_0 = patch_psnrs(subset,clean[:,[0]],to_rgb=True)
    # psnrs_1 = patch_psnrs(subset,clean[:,[1]],to_rgb=True)
    # print("psnrs_0.shape: ",psnrs_0.shape)
    # psnrs_0 = np.sort(psnrs_0,1)

    # -- at index 0 --
    ave_0 = np.mean(psnrs_0,0)[:nN].ravel()
    std_0 = np.std(psnrs_0,0)[:nN].ravel()
    log += "@ index = 0\n"
    log += str(psnrs_0[:nB,:nN]) + "\n"
    log += "ave: " + str(ave_0) + "\n"
    log += "std: " + str(std_0) + "\n"

    # -- at index 1 --
    # ave_1 = np.mean(psnrs_1,1)[:nB].ravel()
    # std_1 = np.std(psnrs_1,1)[:nB].ravel()
    # log += "@ index = 1\n"
    # log += str(psnrs_1[:nB,:nN]) + "\n"
    # log += "ave: " + str(ave_1) + "\n"
    # log += "std: " + str(std_1) + "\n"

    return log


# def create
