# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:11:55 2019

@author: Administrator
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

###############################################################################
# user defined function
###############################################################################


def denormalized_data(normalized_dataset, scale_norm, DATA_ORGANIZATION, data_type):
    
    recovered_dataset = []
    
    dict = ['correlation_coeff_max', 'correlation_coeff_min', 'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min']
    
    if DATA_ORGANIZATION == 'no combination':

        for i in range(np.shape(normalized_dataset)[0]):
            
            temp_data_type = data_type[i]
            temp_recovered_data = normalized_dataset[i] * (scale_norm[dict[2 * temp_data_type]] - scale_norm[dict[2 * temp_data_type + 1]]) \
            + scale_norm[dict[2 * temp_data_type + 1]]
            recovered_dataset.append(temp_recovered_data)
        
        recovered_dataset = np.array(recovered_dataset)        
        
    elif DATA_ORGANIZATION == 'partial combination':
        
        data_type = [0, 1, 2]
        
        for i in range(np.shape(normalized_dataset)[0]):
            for k in range(len(data_type)):
                temp_data_type = data_type[k]               
                temp_recovered_data = normalized_dataset[i,:, k] * (scale_norm[dict[2 * temp_data_type]] - scale_norm[dict[2 * temp_data_type + 1]]) \
                + scale_norm[dict[2 * temp_data_type + 1]]
            recovered_dataset.append(temp_recovered_data)
                
        recovered_dataset = np.array(recovered_dataset)        
        
    elif DATA_ORGANIZATION == 'full combination':
        
        data_type = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        
        for i in range(np.shape(normalized_dataset)[0]):
            for k in range(len(data_type)):
                temp_data_type = data_type[k]               
                temp_recovered_data = normalized_dataset[i,:, k] * (scale_norm[dict[2 * temp_data_type]] - scale_norm[dict[2 * temp_data_type + 1]]) \
                + scale_norm[dict[2 * temp_data_type + 1]]
            recovered_dataset.append(temp_recovered_data)
        
        recovered_dataset = np.array(recovered_dataset)
    
    return recovered_dataset