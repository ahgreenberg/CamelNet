#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:35:55 2024

@author: adamgreenberg
"""

import matplotlib.pyplot as plt
import numpy as np
import CamelData as cd

def show_examples(ae, num_example = 3, std_background = 1, 
                      npt_proj = 128,  save_fig = True):


    data_test = cd.CamelDataset( *cd.generate_camels(num_camel=num_example, 
                                                     locations_per_camel=1,
                                                     hump_weights=[0,15],
                                                     std_background = std_background) )
    
    fig = plt.figure(2, figsize = (9,6.75))
    fig.clf()
    for i in range(num_example):
        ax = fig.add_subplot(num_example,1,i+1)
        
        camel_truth, camel_corrupt = data_test[i]
        camel_recon,encoding = ae(camel_corrupt.view(1,1,-1))

        pl1, = ax.plot(camel_truth.detach().numpy().flatten(), ls='--', c='b')
        pl2  = ax.scatter(range(len(camel_corrupt)),camel_corrupt.detach().numpy().flatten(), 
                         s=15, c='g', marker='x')
        
        pl3, = ax.plot(camel_recon.detach().numpy().flatten(), c = 'r')        
        ax.set_title(f"Example {i+1:d}")
        ax.set_xlim(0,255)
        ax.grid()
        
        if not i: ax.legend([pl1, pl2, pl3], ["Pure signal, $y(t)$", "Noisy shifted signal, $y'(t)\equiv y(t+\Delta t)+\epsilon(t)$", "$F^{-1}(F(y'(t)))$"], fontsize=9)
        
    fig.suptitle(f'Example reconstructions, background={std_background:.1f}')  
    fig.tight_layout()

    if save_fig: fig.savefig(f'plots/example_recons_noise={std_background:.0f}.png')
    
    data_proj = cd.CamelDataset( *cd.generate_camels(num_camel=3,
                                                    locations_per_camel=npt_proj) )
    
    _,camel_corrupt = data_proj[:]
    
    _,encodings = ae(camel_corrupt)
    encodings = encodings.detach().numpy()
        
    dims = np.random.choice(encodings.shape[1], 4, replace=False)
    
    fig = plt.figure(3, figsize = (6.4, 4.8))
    fig.clf()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
        
    for i in range(num_example):
        ax1.scatter(encodings[ i*npt_proj:(i+1)*npt_proj,dims[0]], 
                    encodings[ i*npt_proj:(i+1)*npt_proj,dims[1]] )
        ax2.scatter(encodings[ i*npt_proj:(i+1)*npt_proj,dims[2]], 
                    encodings[ i*npt_proj:(i+1)*npt_proj,dims[3]] )
    
    ax1.set_xlabel(f'Dimension {dims[0]:.0f}')
    ax1.set_ylabel(f'Dimension {dims[1]:.0f}')
    ax2.set_xlabel(f'Dimension {dims[2]:.0f}')
    ax2.set_ylabel(f'Dimension {dims[3]:.0f}')
    fig.suptitle("Example projected ensembles, varying shift and noise")
    
    r = np.diff( ax1.get_xlim() )/np.diff(ax1.get_ylim() )
    ax1.set_aspect(r)
    ax1.grid()
    
    r = np.diff( ax2.get_xlim() )/np.diff(ax2.get_ylim() )
    ax2.set_aspect(r)
    ax2.grid()
    
    if save_fig: fig.savefig(f'plots/example_projs_noise={std_background:.0f}.png')