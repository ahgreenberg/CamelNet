#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:48:23 2024

@author: adamgreenberg
"""
import torch as tc
import CamelData as cd
import numpy as np
import CamelConvNet as ccn

def loss_function(truth, reconstructed, encodings, num_fold):

    # base loss is mean squared error    
    loss_mse = tc.nn.MSELoss(reduction = 'sum')(truth, reconstructed)
            
    if num_fold<2: return loss_mse, 0
    
    # calculate number of indepedent realizations of same camel
    num_ind = int(encodings.shape[0]/num_fold)
    
    # fold encodings into 3D such that identical camels share a dimension
    encodings_cube = encodings.view(num_ind, num_fold, encodings.shape[-1])
    
    # calculate spread and mean per camel per encoded dimension
    stds = tc.std(encodings_cube, dim=1)
    means = tc.mean(encodings_cube, dim=1)
    
    # calculate max spread across all encoded dimensions, per camel
    stds_max = tc.max( stds, dim=1)[0]
    
    # define contractive loss as the mean spread across all camels, normalized
    # by each camel's mean vector length in the encoded space
    means_norm = tc.linalg.norm( means, ord = 2, dim=1)
    loss_contractive = tc.mean( (stds_max/means_norm)**2 )
    
    return loss_mse, loss_contractive

def train( state_file = None,    num_epoch = 16,  
           num_camel = 4096,     locations_per_camel = 8, 
           batch_size = 64,      lam = 1E4, 
           learning_rate = 3E-4, show_progress = True):
    
    # instantiate autoencoder and preload weights, if applicable
    ae = ccn.CamelAutoEncoder()
    if state_file is not None: ae.load_state_dict( tc.load(state_file) )
    
    if not num_epoch: return ae, None, None

    # initialize dataset        
    dataset = cd.CamelDataset( *cd.generate_camels( num_camel = num_camel, 
                                locations_per_camel = locations_per_camel) ) 
    data_loader = tc.utils.data.DataLoader( dataset, 
                                            batch_size = batch_size, 
                                            shuffle = False)
    
    msg = "Locations per camel must divide data samples per batch"
    assert (not data_loader.batch_size % locations_per_camel), msg
    
    # initialize optimizer with model's parameters
    optimizer = tc.optim.Adam( ae.parameters(), lr = learning_rate )
    
    # save both components of loss function
    loss_curves = np.zeros([num_epoch, 2])
    
    # tell model training is beginning
    ae.train()
    for i in range(num_epoch):
                        
        for features_clean,features_corrupt in data_loader:
            
            optimizer.zero_grad()
            
            # perform forward-propagation and get components of the loss
            features_recon, encodings = ae(features_corrupt)
            loss_base, loss_contractive = loss_function( features_clean, 
                                                         features_recon, 
                                                         encodings, 
                                                         locations_per_camel)
            
            # calculate total loss and perform back-propagation
            loss = loss_base + lam*loss_contractive
            loss.backward()
            optimizer.step()
            
            # save loss for current iteration of this epoch
            loss_curves[i,0] += loss_base
            loss_curves[i,1] += lam*loss_contractive
            
        # normalize loss for all iterations in this epoch    
        loss_curves[i,:] /= len(dataset)
        
        if show_progress:
            print(f"Epoch: {i:2d}, ", end="")
            print(f"Base loss: {loss_curves[i,0]:.3f}, ", end="")
            print(f"Contractive loss: {loss_curves[i,1]:.3f}")

    ae.eval()
    
    return ae, dataset, loss_curves