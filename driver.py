#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 00:35:55 2024

@author: adamgreenberg
"""

import sys
sys.path.append('./classes/')

import torch as tc
import CamelTrainer as ct
import CamelPlotter as cpl

state_file_in = None #"state_dictionaries/states_nominal_deconv.pkl"
state_file_out = None

ae, data_train, loss_curves = ct.train( num_epoch = 16,
                                        state_file = state_file_in )

if state_file_out is not None: tc.save(ae.state_dict(), state_file_out)

cpl.show_examples(ae, std_background = 5, save_fig = False)
