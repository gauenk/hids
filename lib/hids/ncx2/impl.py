
# -- python --
from easydict import EasyDict as edict

# -- data frame --
import pandas as pd

# -- linalg --
import numpy as np
import torch as th

# -- plotting --
import matplotlib.pyplot as plt

# -- import testing --
from hids.testing import get_exp_mesh

FONTSIZE = 12

def get_plot_mesh():

    # -- plotting grids --
    plots = edict()
    labels = edict()
    ticks = edict()

    # -- ref number --
    rnum_grid = [1,2,3,4,5]#,10]
    plots.rnum = [1,3,5]

    # -- noise intensities --
    sigma_grid = np.arange(10.,50.+1.,1.)/255.
    # sigma_grid[0] = 5./255.
    sigma2_grid = sigma_grid**2
    plots.sigma2 = sigma2_grid
    # ticks.sigma2 = sigma_grid[

    # -- delta grid --
    delta_grid = np.arange(0,100.+1,1.)
    delta_grid[0] = 0.01
    # delta_grid = [0,1e-4,1e-3,1e-2,1e-1,1.]
    plots.delta = [0,1e-3,1.]

    # -- dim grid --
    dim_grid = [98]
    plots.dim = [98]

    # -- seeds --
    seeds = [-1]#123,234,345,]

    # -- compute meshgrid --
    fields = {'rnum':rnum_grid,'sigma2':sigma2_grid,
              'dim':dim_grid,'delta':delta_grid,'seed':seeds}

    # -- create exp mesh --
    exps = get_exp_mesh(fields)

    return exps,plots

def compute_snr_on_point(point):

    # -- unpack --
    D = point['dim']
    alpha = (point['rnum']+1)/point['rnum']
    s2 = point['sigma2']
    delta = point['delta']

    # -- eval --
    val = (D * alpha**2 * s2**2 + alpha * s2 * delta)
    val /= (2 * D * alpha * s2 + 4 * delta )**(0.5)
    return val

def compute_snr_on_grid(grid):
    for point in grid:
        point['snr'] = compute_snr_on_point(point)

def create_grid_plots():

    # -- create experiments --
    grid,plots = get_plot_mesh()
    compute_snr_on_grid(grid)
    df = pd.DataFrame(grid)
    print(df)

    # -- get color levels --
    print(df['snr'].min(),df['snr'].max())
    # snr = df['snr'].to_numpy()
    snr = np.log(df['snr'].to_numpy())
    # snr = np.log10(df['snr'].to_numpy())
    levels = np.linspace(snr.min(),snr.max(),30)
    print(levels)

    # -- inspect --
    print(df[df['delta'].between(.9,1.1)][df['rnum'].between(2.5,3.5)]['snr'])
    print(df[df['delta'].between(9.9,10.1)][df['rnum'].between(2.5,3.5)]['snr'])
    print(df[df['delta'].between(97.9,98.1)][df['rnum'].between(2.5,3.5)]['snr'])

    # -- [a 2 x 3 grid of plots] --
    # fig,ax = plt.subplots(2,3,figsize=(8,4))
    # plot_pair(df,ax[0,0],fix=['rnum',1],strat=['sigma2','delta'],
    #           levels=levels,is_top=True,is_left=True)
    # plot_pair(df,ax[0,1],fix=['rnum',3],strat=['sigma2','delta'],
    #           levels=levels,is_top=True,is_left=False)
    # plot_pair(df,ax[0,2],fix=['rnum',5],strat=['sigma2','delta'],
    #           levels=levels,is_top=True,is_left=False)
    # plot_pair(df,ax[1,0],fix=['delta',0.01],strat=['sigma2','rnum'],
    #           levels=levels,is_top=False,is_left=True)
    # plot_pair(df,ax[1,1],fix=['delta',10.],strat=['sigma2','rnum'],
    #           levels=levels,is_top=False,is_left=False)
    # plot_pair(df,ax[1,2],fix=['delta',100.],strat=['sigma2','rnum'],
    #           levels=levels,is_top=False,is_left=False)
    # plt.savefig("./ncx2.png",dpi=500,transparent=True,bbox_inches='tight')

    # -- [a 1 x 4 grid of plots] --
    fig,ax = plt.subplots(1,2,figsize=(4,2))
    plot_pair(df,ax[0],fix=['rnum',1],strat=['sigma2','delta'],
              levels=levels,is_top=False,is_left=True)
    plot_pair(df,ax[1],fix=['rnum',5],strat=['sigma2','delta'],
              levels=levels,is_top=False,is_left=False)
    plt.savefig("./ncx2_a.png",dpi=500,transparent=True,bbox_inches='tight')
    plt.close("all")

    fig,ax = plt.subplots(1,2,figsize=(4,2))
    plot_pair(df,ax[0],fix=['delta',10.],strat=['sigma2','rnum'],
              levels=levels,is_top=False,is_left=True)
    plot_pair(df,ax[1],fix=['delta',100.],strat=['sigma2','rnum'],
              levels=levels,is_top=False,is_left=False)
    plt.savefig("./ncx2_b.png",dpi=500,transparent=True,bbox_inches='tight')


    plt.close("all")

def plot_pair(df,ax,fix,strat,levels,is_top=False,is_left=False):

    key,val = fix[0],fix[1]
    # -- fix field --
    tol = 1e-8
    df = df[df[key].between(val-tol,val+tol)]
    print(len(df))

    # -- remove dups of "strat" --
    strat += ['snr']
    print(strat)
    df = df[strat].drop_duplicates()

    # -- pivot to create mesh--
    df = df.pivot(strat[0],strat[1])

    # -- old plot --
    # x = df[strat[0]].to_numpy()
    # y = df[strat[1]].to_numpy()
    # z = df[strat[2]].to_numpy()

    # -- plotting --
    X = df.columns.levels[1].values
    Y = df.index.values
    Z = df.values

    # -- mesh --
    Xi,Yi = np.meshgrid(X, Y)

    # -- apply transform --
    xtrans = get_field_trans(strat[1])
    Xi = xtrans(Xi)
    ytrans = get_field_trans(strat[0])
    Yi = ytrans(Yi)

    # -- plot! [finally, lol] --
    tol = 1e-10
    Z = np.log(Z+tol)
    # Z = np.log10(Z+tol)
    print(Z.min(),Z.max())
    # vmin,vmax=0,0.405 # null
    vmin,vmax=-7.11,-0.89 # log e
    # vmin,vmax=-3.8,-0.38 # log 10
    ax.contourf(Yi, Xi, Z, alpha=0.7, cmap=plt.cm.binary,
                levels=levels,vmin=vmin,vmax=vmax)

    # -- subtitle --
    title = subtitle(fix[0],fix[1])
    ax.set_title(title,fontsize=FONTSIZE)

    # -- translate axis-labels --
    xtick_label,xtick_locs = get_tickinfo(strat[0])
    ytick_label,ytick_locs = get_tickinfo(strat[1])
    xlabel = translate_label(strat[0])
    ylabel = translate_label(strat[1])

    if not(is_top):
        ax.set_xlabel(xlabel,fontsize=FONTSIZE)
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_label)
    else:
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels([])

    if is_left:
        ax.set_ylabel(ylabel,fontsize=FONTSIZE)
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_label)
    else:
        ax.set_yticks(ytick_locs)
        ax.set_yticklabels([])


def subtitle(field,val):
    if field == "delta":
        if val < 1:
            print(val)
            return "$\Delta$ = "+"%1.0e"%val
        else:
            return "$\Delta$ = "+"%d"%int(val)
    elif field == "sigma2":
        return "$\sigma$ = "+"%2.2f"%val
    elif field == "rnum":
        return "N = "+"%d"%val
    else:
        raise ValueError(f"subtitle failed [{field}]")

def get_field_trans(field):
    def null(x): return x
    if field == "delta":
        return np.log10
    elif field == 'sigma2':
        return np.sqrt
    elif field == "rnum":
        return null
    else:
        raise ValueError(f"ticklabels failed [{field}]")
    return fields

def get_tickinfo(field):
    if field == "delta":
        # labels = ['0.','0.5','1.','1.5']
        # locs = [0.,0.5,1.,1.5]
        # labels = ['0.','1.','1.5','2.']
        # locs = [0.,0.5,1.5,2.]
        labels = ['1e-2', '1e-1', '1.','10','100']
        locs = list(np.log10(np.array([0.01, .1, 1.,10.,100.])))
        return labels,locs
    elif field == "sigma2":
        labels = ['10','20','30','40','50']
        # locs = [(10./255.)**2,(20./255.)**2,(30./255.)**2,
        #         (40./255.)**2,(50./255.)**2,]
        locs = [(10./255.),(20./255.),(30./255.),
                (40./255.),(50./255.),]
        return labels,locs
    elif field == "rnum":
        labels = ['1','2','3','4','5']
        locs = [1.,2.,3.,4.,5.]
        return labels,locs
    else:
        raise ValueError(f"ticklabels failed [{field}]")

def translate_label(field):
    if field == "delta":
        return "$\Delta$"
        # return "log10($\Delta$)"
    elif field == "sigma2":
        return "$\sigma$"
    elif field == "rnum":
        return "N"
    else:
        raise ValueError(f"translation failed [{field}]")


