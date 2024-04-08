#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:41:29 2024

@author: adamgreenberg
"""
import numpy as np
import torch as tc

class CamelDataset(tc.utils.data.Dataset):
    
    num_samp : int
    locations_per_camel : int
    
    def __init__(self, ys, ys_noise, locations_per_camel):
        super().__init__()
        
        self.ys = tc.Tensor(ys)
        self.ys_noise = tc.Tensor(ys_noise)
        
        self.num_samp = ys.shape[1]
        self.locations_per_camel = locations_per_camel
        
    def __len__(self):
        return self.ys.shape[0]
    
    def __getitem__(self, ind):
        return (self.ys[ind,:],
                self.ys_noise[ind,:] )

def generate_camels( std_background = 1,
                     data_extent = [-128, 128],
                     num_camel = 8192,
                     locations_per_camel = 4,
                     hump_per_camel = 8,
                     camel_locations = [-64, 64],
                     hump_locations = [-32, 32], 
                     hump_widths = [2, 10],
                     hump_weights = [0, 15]):
    
    xs = np.arange(*data_extent)
    ys_truth = np.zeros([num_camel*locations_per_camel, xs.size])
    ys_corrupt = np.random.normal(0, std_background, ys_truth.shape)

    
    hump_fun = lambda w,m,s: w*np.exp( -0.5 * ((xs-m)/s)**2 )
    
    U = np.random.uniform
    for icamel in range(num_camel):
        weights = np.array([U(*hump_weights) for _ in range(hump_per_camel)])
        means = np.array([U(*hump_locations) for _ in range(hump_per_camel)])
        stds = np.array([U(*hump_widths) for _ in range(hump_per_camel)])
        
        for iloc in range(locations_per_camel):
            
            loc = U(*camel_locations)
            iy = iloc+locations_per_camel*icamel
            
            for weight,mean,std in zip(weights, means, stds):
                ys_truth[iy,:] += hump_fun(weight, mean, std)
                ys_corrupt[iy,:] += hump_fun(weight, mean+loc, std)
                        
            ys_truth[iy,:] /= np.max(ys_truth[iy,:])
            ys_corrupt[iy,:] /= np.max(ys_corrupt[iy,:])
            
            y_cumsum = np.cumsum(ys_truth[iy,:])
            ishifts = np.nonzero(y_cumsum >= y_cumsum[-1]/2)
            
            # ipeak = np.argmax(ys_truth[iy,:])
            ys_truth[iy,:] = np.roll(ys_truth[iy,:], 128-ishifts[0][0])
            
      
    return ys_truth, ys_corrupt, locations_per_camel