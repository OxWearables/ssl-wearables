import os
from os.path import expanduser
home = expanduser("~")

import numpy as np
#import bottleneck as bn
import pandas as pd

#PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm.auto import tqdm
import gc
import time

from sklearn import metrics

#Captum & LRP
#import sys
#sys.path
#assuming you've got captum installed somewhere...?
#sys.path.append(home + '/scripts/python/proprietary/captum/')

from pathlib import Path 

from captum.attr import LRP, IntegratedGradients
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule, IdentityRule

from matplotlib.colors import ListedColormap

cuda = torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
cudnn.benchmark = True

def set_seed():
    # For reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True
    if cuda:
        torch.cuda.manual_seed_all(random_seed)

'''
# Grab a GPU if there is one
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} GPU device: {} ({})".format(device, torch.cuda.current_device(), torch.cuda.get_device_name()))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
else:
    device = torch.device("cpu")
    print("Using {}".format(device))
'''
set_seed()

#----------------------------------------------------------------------#
#------------------------Attribution-----------------------------------#
#______________________________________________________________________#
class Attribute(object):

    def __init__(self, model, analysis_model, normalise=False, relevance_clip=False, relevance_clip_plim=98, **kwargs):
        self.model=model
        self.analysis_model=analysis_model
        self.normalise=normalise
        self.relevance_clip=relevance_clip
        self.relevance_clip_plim=relevance_clip_plim
        self.kwargs=kwargs

    def __call__(self, input):
        attribution=_attribute(input, self.model, self.analysis_model, **self.kwargs)   

        if self.relevance_clip:
            attribution=_relevance_clip(attribution, plim=self.relevance_clip_plim)
        if self.normalise:
           attribution= _normalise_attribution(attribution)

        return attribution

def _attribute(input, model, analysis_model, **kwargs):
    model.zero_grad()   
    attribution = analysis_model.attribute(input, **kwargs)
    return attribution

def _normalise_attribution(attribution):
    attribution=attribution/np.max(np.abs(attribution.reshape(-1)))
    return attribution

def _relevance_clip(attribution, plim=98):
    Rclip=attribution
    if torch.is_tensor(attribution):
        if plim>1: plim=plim/100
        Rclip[attribution>torch.quantile(attribution, plim)]=torch.quantile(attribution, plim)
        Rclip[attribution<torch.quantile(attribution, 1-plim)]=torch.quantile(attribution, 1-plim)
    else:
        if plim<1: plim=plim*100
        Rclip[attribution>np.percentile(attribution, plim)]=np.percentile(attribution, plim)
        Rclip[attribution<np.percentile(attribution, 100-plim)]=np.percentile(attribution, 100-plim)
    return Rclip
#----------------------------------------------------------------------#
#------------------- Visualisation Utils ------------------------------#
#______________________________________________________________________#    
import matplotlib
from scipy import signal
import matplotlib.pyplot as plt
#import pywt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

def tickLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

def compute_cwt(x, time, fs, transformation=None, pad_pc=0.1, nsample=1000):
    
    #normalise signal between -1:-1: to compare between CWT plots
    if transformation is not None:
        x=transformation(x)

    dt = 1/fs  # fs Hz sampling
    w = 5 #Omega0
    freq = np.linspace(0.5, fs/2, nsample) 

    #fundamental frequency: f = w*fs / (2*s*np.pi) 
    #-> where s is the wavelet width parameter.
    #Similarly we can get the wavelet width parameter at f: s = w*fs / (2*f*np.pi)

    widths = w*fs / (2*freq*np.pi) 

    #'''n.b. pad the signal to remove edge effects'''
    pad_width=int(np.floor((len(x)*pad_pc)))
    #also see: https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
    #https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad
    #x=np.pad(sig, pad_width=pad_width, mode='constant', constant_values=0)
    x_pad=np.pad(x, pad_width=pad_width, mode='reflect')
    time_pad=np.pad(time,  pad_width=pad_width, mode='edge')

    cwtm = signal.cwt(x_pad, signal.morlet2, widths,w=w)
    cwt_abs=np.abs(cwtm)

    return cwt_abs, time_pad, freq

"""Plotting Utils"""
def _plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

def _colormap(x, darken=1, **kwargs):
    #NOTE define your color mapping function here.
    return _firered(x, darken, **kwargs)


def _firered(x, darken=1.0):
    """ a color mapping function, receiving floats in [-1,1] and returning
        rgb color arrays from (-1 ) lightblue over blue, black (center = 0), red to brightred (1)
        x : float array  in [-1,1], assuming 0 = center
        darken: float in ]0,1]. multiplicative factor for darkening the color spectrum a bit
    """

    #print 'color mapping {} shaped into array to "firered" color spectrum'.format(x.shape)
    assert darken > 0
    assert darken <= 1

    x *= darken

    hrp  = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x-0.00,0,0.25)/0.25
    hgn = np.clip(-x-0.25,0,0.25)/0.25
    hrn = np.clip(-x-0.50,0,0.50)/0.50

    colors =  np.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)
    #print '    returning rgb mask of size {}'.format(colors.shape)
    return colors


def _interpolate_all_samples(Y, fold=10):
    # assumes a N x C matrix Y
    #interpolates along N axis to the fold amount of points.
    N, D = Y.shape
    x = np.arange(N)
    xinterp = np.linspace(0,N-1, num=N*fold)
    Yinterp = np.concatenate([np.interp(xinterp, x, Y[:,d])[:,None] for d in range(D)], axis=1)
    #print Y.shape, xinterp.shape, xinterp[-1], Yinterp.shape
    return xinterp, Yinterp

def create_lrp_colormap(ncolors=256, limits=[-1, 1]):
        
    Rdummpy=np.array([np.linspace(np.min(limits), np.max(limits), num=ncolors, dtype='float')])
    rel_colors = _colormap(Rdummpy)

    #add white parts to ends
    rel_colors;

    vals = np.ones((ncolors, 4))
    vals[:, 0] = rel_colors[:, :, 0]
    vals[:, 1] = rel_colors[:, :, 1]
    vals[:, 2] = rel_colors[:, :, 2]
    newcmp = ListedColormap(vals)
    #_plot_examples([cm.get_cmap('seismic', ncolors), newcmp])

    return newcmp

"""LRP visusalisation utils"""
def _smooth_attribution(attribution, window_size=5):
    attribution=pd.DataFrame(attribution).rolling(window_size, center=True, min_periods=1, axis=0).mean().to_numpy()
    return _normalise_attribution(attribution)

def plot_raw_accel(handle, X, t, **parm):
    axis=handle.plot(t, X)
    #plt.legend(iter(axis), ('x', 'y', 'z'),fontsize=12)
    handle.set_ylim(parm['ylim'])
    return (handle, axis)

def plot_cwt(handle, t_cwt, freq_cwt, cwt_abs, **parm):

    cwt_axis=handle.pcolormesh(t_cwt, freq_cwt, cwt_abs, cmap='jet', shading='gouraud', vmin=0, vmax=parm['cwt_vmax'])
    handle.set_yscale('log')
    handle.set_yticks(parm['cwt_yticks'])
    handle.set_ylim(parm['cwt_ylim'])
    handle.yaxis.set_major_formatter(ticker.FuncFormatter(tickLogFormat))

    handle.tick_params('both', length=5, width=1, which='major')
    handle.tick_params('both', length=3, width=1, which='minor')
    return (handle, cwt_axis)

def plot_cwt_cbar(fig, cwt_axis, axis=[0.91, 0.395, 0.01, 0.22],  orientation='vertical', **parm):
    cwt_cbar_ax = fig.add_axes(axis)
    cwt_colorbar=plt.colorbar(cwt_axis, cax=cwt_cbar_ax, orientation =orientation)
    #add colorbar limits
    cwt_axis.set_clim(parm['cwt_cbar_limit'])
    #add colorbar ticks
    cwt_colorbar.set_ticks(parm['cwt_cbar_ticks'])
    cwt_colorbar.set_label(parm['cwt_label']) 
 
    return (fig, cwt_colorbar)

def plot_pcolor_attribution(handle, attribution, x, **parm):

    yv = np.arange(0, 5, 1)
    x, y = np.meshgrid(x, yv)
    z=np.concatenate((np.full([attribution.shape[0], 1], np.nan), attribution, np.full([attribution.shape[0], 1], np.nan)), 1).transpose()
    axis = handle.pcolormesh(x, y, z, cmap=parm['lrpmap'], shading='nearest',vmin=parm['lrp_limits'][0], vmax=parm['lrp_limits'][1])
    handle.set_ylim([0.5, 3.5])
    handle.set_yticks([1, 2, 3])
    handle.set_yticklabels(parm['channel_labels'])
    return (handle, axis)

def add_lrp_colorbar(fig, axis=[0.35, -0.001, 0.33, 0.02], orientation='horizontal', **parm):        
    lrp_cbar_ax = fig.add_axes(axis)
    #lrp_cbar_ax.axes.get_xaxis().set_visible(False)
    #lrp_cbar_ax.axes.get_yaxis().set_visible(False)
    #and create another colorbar with:
    cbar=matplotlib.colorbar.ColorbarBase(lrp_cbar_ax, cmap=parm['lrpmap'], orientation = orientation)
    cbar.set_label('LRP')
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels(['-1','', '0','', '1'])
    return (fig, cbar)

def plot_scatter_attribution(handle, attribution, x, maxamp=1, gap=1.25, **parm):
    
    x, attribution = _interpolate_all_samples(attribution, fold=50)
    nrel = attribution / np.max(np.abs(attribution))
    rel_colors = _colormap(nrel)
    t_interp=np.linspace(parm['sec_start'], parm['sec_end'], len(x))

    rels=attribution
    ch_ticks=np.arange(parm['nchannels'])*gap
    axis=[]
    for j in range(parm['nchannels']):

            rel = rels[:,j]
            rel_colors_=rel_colors[:,j,:]

            maxamp = max(maxamp, np.abs(rel).max())

            ax=handle.scatter(x=t_interp,
                                y=rel+ch_ticks[j],
                                c=rel_colors_,
                                s=2,
                                marker = '.',
                                edgecolor='none')
            axis.append(ax)
            maxrelpos = np.argmax(rel)
            #handle.scatter(x[maxrelpos], rel[maxrelpos],marker = 'o',c = 'none',edgecolor='r',linewidth=0.5)
            #handle.plot(x[maxrelpos], rel[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,j,:]
            
    #print maxamp
    maxamp *= 1.1
    handle.set_ylim([ch_ticks.min()-gap*1.01, ch_ticks.max()+gap*1.01])
    handle.set_xticks(parm['xticks'])

    handle.set_yticks(ch_ticks) #show single vertical grid line thresholding zero
    handle.set_yticklabels(parm['channel_labels']) #deactivate tick label texts (no info gained from this)#show
    handle.grid(True, linewidth=0.5, axis='x')
    handle.set_axisbelow(True) # grid lines behind all other graph elements
    handle.set_xlim(parm['xlim'])

    return (handle, axis)    

def overlay_scatter_attribution(handle, attribution, values, x, maxamp=1,overlay_scatter_sz=2, **parm):
    
    #interpolate measurements and colorize
    x, rels = _interpolate_all_samples(attribution, fold=50)
    _, vals = _interpolate_all_samples(values, fold=50)

    x = np.repeat(x[...,None] ,vals.shape[1], axis=1).flatten()
    vals = vals.flatten()
    rels_=rels.flatten()

    nrel = rels / np.max(np.abs(rels))
    rel_colors = np.reshape(_colormap(nrel) , [-1, 3])

    #optimize z-order for drawing: highlight relevances with high magnitudes
    I = np.argsort(np.abs(nrel.flatten()))
    vals = vals[I]
    x=x[I]
    rel_colors=rel_colors[I,:]

    t_interp=np.linspace(parm['sec_start'], parm['sec_end'], len(x))
    t_interp=t_interp[I]

    axis=handle.scatter(x=t_interp,
                    y=vals,
                    c=rel_colors,
                    s=overlay_scatter_sz,
                    marker = '.',
                    edgecolor='none')

    handle.set_ylim(parm['ylim'])
    #handle.set_ylim([-maxamp, maxamp])
    handle.set_xticks(parm['xticks'])
    handle.set_yticks([-maxamp, 0, maxamp])
    ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
    ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
    handle.set_yticklabels(ticklabs)

    #ax.yaxis.tick_right()
    handle.set_xlim(parm['xlim'])
    handle.set_xticks(parm['xticks'])
    handle.grid(True, linewidth=0.5, axis='x')
    handle.set_axisbelow(True) # grid lines behind all other graph elements
    
    return (handle, axis)

def add_zoom_patch(handle, sec_start, sec_end, box_height=1.95):
    sec_length=sec_end-sec_start
    handle.add_patch(
        patches.Rectangle(
            xy=(sec_start, -0.95),  # point of origin.
            width=sec_length, height=box_height, linewidth=1.5,
            color='black', fill=False))
    return handle


#----------------------------------------------------------------------#
#---------------------Attribution Utils--------------------------------#
#______________________________________________________________________#

def forward_by_batches(model, data_generator, loss_fn=None, device='cpu'):
    ''' Forward pass model on a dataset. 
    Do this by batches so that we don't blow up the memory. '''
    model.eval()
    current_loss=0
    Y = []
    Yfit = []
    posterior=[]  
    softmax = nn.Softmax(dim=1)
    #Forward pass
    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        for v, data in enumerate(data_generator):
            X_batch, Y_batch = data
            # Transfer to GPU/CPU
            X_batch, Y_batch = X_batch.to(device, dtype=torch.float), Y_batch.to(device,  dtype=torch.long)
            #evaluate model on the validation set
            logits = model(X_batch)
            prob = softmax(logits)     
            Y_batch = torch.argmax(Y_batch, dim=1).view(-1)
            if loss_fn is not None:
                loss=loss_fn(logits, Y_batch)
                current_loss += loss.item()

            Yfit.extend(torch.max(logits, 1)[1].cpu().detach().numpy())
            Y.extend(Y_batch.cpu().detach().numpy())
            posterior.extend(prob.cpu().detach().numpy())
        current_loss = current_loss / len(data_generator) 
        Yfit = np.stack(Yfit)
        Y = np.stack(Y)
        posterior=np.stack(posterior)
    return Y, Yfit, current_loss, posterior