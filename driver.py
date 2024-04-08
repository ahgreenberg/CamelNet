#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 00:35:55 2024

@author: adamgreenberg
"""
import numpy as np
import torch as tc
import CamelData as cd
import matplotlib.pyplot as plt
import CamelConvNet as ccn

#%%
state_file_in = "state_dictionaries/states_nominal.pkl"
state_file_out = None

ae, data_train, loss_curves = ccn.train( num_epoch = 0, 
                                         state_file = state_file_in )

if state_file_out is not None:
    tc.save(ae.state_dict(), state_file_out)

#%%
if loss_curves is not None:
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot()
    ax.plot(range(1, loss_curves.shape[0]+1), np.sum(loss_curves, axis=1), c='b')
    
    ax2 = ax.twinx()
    ax2.plot(range(1, loss_curves.shape[0]+1), loss_curves[:,1], c='r')
    
    fig.savefig('plots/loss_curve.png')

#%%
num_example = 3
std_background = 1
npt_proj = 256

data_test = cd.CamelDataset( *cd.generate_camels(num_camel=num_example, 
                                                 locations_per_camel=1,
                                                 hump_weights=[0,10],
                                                 std_background = std_background) )

fig = plt.figure(2)
fig.clf()
for i in range(num_example):
    ax = fig.add_subplot(num_example,1,i+1)
    
    camel_truth, camel_corrupt = data_test[i]
    camel_recon,encoding = ae(camel_corrupt.view(1,1,-1))
    
    ax.plot(camel_truth.detach().numpy().flatten(), ls='--', c='b')
    ax.plot(camel_recon.detach().numpy().flatten(), c = 'r')
    ax.scatter(range(len(camel_corrupt)),camel_corrupt.detach().numpy().flatten(), s=1, c='g')

fig.suptitle(f'Example reconstructions, background={std_background:.1f}')    
fig.savefig(f'plots/example_recons_noise={std_background:.0f}.png')

data_proj = cd.CamelDataset( *cd.generate_camels(num_camel=3,
                                                locations_per_camel=npt_proj) )

_,camel_corrupt = data_proj[:]

_,encodings = ae(camel_corrupt)
encodings = encodings.detach().numpy()
    
dims = np.random.choice(encodings.shape[1], 4, replace=False)

fig = plt.figure(3)
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
fig.suptitle("Example ensembles, varying shift and noise")

r = np.diff( ax1.get_xlim() )/np.diff(ax1.get_ylim() )
ax1.set_aspect(r)
ax1.grid()

r = np.diff( ax2.get_xlim() )/np.diff(ax2.get_ylim() )
ax2.set_aspect(r)
ax2.grid()

fig.savefig(f'plots/example_projs_noise={std_background:.0f}.png')