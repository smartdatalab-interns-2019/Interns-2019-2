# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:11:55 2019

@author: Administrator
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import os.path
import pickle
import create_data_for_deeplearning
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data

###############################################################################
# user defined function
###############################################################################


def load_dataset(filename, if_save_dataset):

    if os.path.isfile(filename):
        
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle) 
            
        data_T = dataset['data']
        label_T =  dataset['label']
        Tag = dataset['tag']
        timestamp_T =  dataset['timestamp']
        n_tag_0 =  dataset['tag0']
        n_tag_1 = dataset['tag1']
        scale_norm = dataset['scale_norm']
        data_type = dataset['data_type']
        
    else:
        print("*"*50)
        print("start to create dataset")
        print("*"*50)
        print("\n")
        
        file1 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_no_mass.pickle'
        file2 = 'F:/Kang/Data/plate_ultrasonic_dataset_197_damage.pickle'
        
        data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
        create_data_for_deeplearning.create_dataset(DATA_MODE, file1, file2, N_FILE_TYPE)
        
        dataset = {'data':data_T, 'label':label_T, 'tag':Tag, 'timestamp':timestamp_T, \
                   'tag0':n_tag_0, 'tag1':n_tag_1, 'scale_norm': scale_norm, 'data_type': data_type}
        
        if if_save_dataset:
            with open(filename, 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type

def create_no_mass_index(data_mode, n_tag_0, n_tag_1):
    
    if data_mode == 'predict_input':
        no_mass_index = np.concatenate((np.arange(n_tag_0 * 8), \
                                        np.arange(n_tag_0 * 8 + n_tag_1 * 8, n_tag_0 * 9 + n_tag_1 * 8),\
                                        np.arange(n_tag_0 * 9 + n_tag_1 * 9, n_tag_0 * 10 + n_tag_1 * 9)), axis = 0)
    
    elif data_mode == 'predict_temperature':
        no_mass_index = np.arange(n_tag_0 * 8)
        
    elif data_mode == 'predict_humidity':
        no_mass_index = np.arange(n_tag_0 * 8)

    elif data_mode == 'full_combination':
        no_mass_index = np.arange(n_tag_0)
        
    elif data_mode == 'partial_combination':
        no_mass_index = np.arange(n_tag_0 * 8)
        
    elif data_mode == 'no_combination':
        no_mass_index =np.concatenate((np.arange(n_tag_0 * 8), \
                                        np.arange(n_tag_0 * 8 + n_tag_1 * 8, n_tag_0 * 9 + n_tag_1 * 8),\
                                        np.arange(n_tag_0 * 9 + n_tag_1 * 9, n_tag_0 * 10 + n_tag_1 * 9)), axis = 0)

    return no_mass_index

def create_data_for_rnn(data_organization, filename_preprocess_data, SAVE_CREATED_DATA):
   
    
    data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
    load_dataset(filename = filename_preprocess_data, if_save_dataset = SAVE_CREATED_DATA)
        
    
    if data_organization == 'no_combination':
        
        return data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type
        
    elif data_organization == 'full_combination':
                                                            
        temperature_data = data_T[(n_tag_0 * 8 + n_tag_1 * 8): (n_tag_0 * 9 + n_tag_1 * 9), :]            
        humidity_data = data_T[(n_tag_0 * 9 + n_tag_1 * 9): (n_tag_0 * 10 + n_tag_1 * 10), :]
        
        data_T_new = np.concatenate((np.expand_dims(temperature_data, axis = 2), \
                                     np.expand_dims(humidity_data, axis = 2)), axis = 2)
        
        N = 8
        
        for i in range(N):
            
            data_T_new = np.concatenate((data_T_new, np.expand_dims(data_T[i:(n_tag_0 * 8 + n_tag_1 * 8):N], axis = 2)), axis = 2)
                
        Tag_new = Tag[::N]
        timestamp_T_new = timestamp_T[:(n_tag_0 * 8 + n_tag_1 * 8):N]
                
        return data_T_new, data_T_new, Tag_new, timestamp_T_new, n_tag_0, n_tag_1, scale_norm, data_type
        
    elif data_organization == 'partial_combination':
                
        correlation_coeff = data_T[: (n_tag_0 * 8 + n_tag_1 * 8), :]                                                     
        temperature_data = data_T[(n_tag_0 * 8 + n_tag_1 * 8): (n_tag_0 * 9 + n_tag_1 * 9), :]            
        humidity_data = data_T[(n_tag_0 * 9 + n_tag_1 * 9): (n_tag_0 * 10 + n_tag_1 * 10), :]
        
        data_T_new = np.concatenate((np.expand_dims(correlation_coeff, axis = 2), \
                                     np.expand_dims(np.repeat(temperature_data, 8, axis = 0), axis = 2), \
                                     np.expand_dims(np.repeat(humidity_data, 8, axis = 0), axis = 2), ), axis = 2)
        
        Tag_new  = Tag
        timestamp_T_new = timestamp_T[:(n_tag_0 * 8 + n_tag_1 * 8)]
                
    return  data_T_new, data_T_new, Tag_new, timestamp_T_new, n_tag_0, n_tag_1, scale_norm, data_type

###############################################################################
# hyperparameters
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset', conflict_handler = 'resolve')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=True,
                    help='augment')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cpu',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true',
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

Loading_DATA = True
SAVE_CREATED_DATA = False
WITH_MASS_LABEL = False
RNN_DATASET = True
DATA_MODE = 'predict_input'

DATA_ORGANIZATION = 'partial_combination'
N_FILE_TYPE = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###############################################################################
# Load data
###############################################################################
'''
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename,
                                                augment_test_data=args.augment)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData, args.batch_size)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, args.eval_batch_size)
gen_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, 1)
'''

if Loading_DATA:

    filename_preprocess_data = 'F:/Kang/Data/plate_ultrasonic_dataset_197_process_' + DATA_MODE + '.pickle'
        
    data_T, label_T, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type = \
    create_data_for_rnn(DATA_ORGANIZATION, filename_preprocess_data, SAVE_CREATED_DATA)
    
    if WITH_MASS_LABEL:
    
        train_input, validation_input, train_label, validation_label,  data_type_train, data_type_test = \
        train_test_split(data_T, label_T, data_type, test_size = 0.2)
        
    else:
        
        if RNN_DATASET:            
            no_mass_index = create_no_mass_index(data_mode = DATA_ORGANIZATION, n_tag_0 = n_tag_0, n_tag_1 = n_tag_1)
        else:
            no_mass_index = create_no_mass_index(data_mode = DATA_MODE, n_tag_0 = n_tag_0, n_tag_1 = n_tag_1)
            
        train_input, validation_input, train_label, validation_label,  data_type_train, data_type_test = \
        train_test_split(data_T[no_mass_index,:], label_T[no_mass_index,:], data_type[no_mass_index], test_size = 0.2)        
    
    train_input = torch.from_numpy(train_input)
    train_label = torch.from_numpy(train_label)       
    train_data = torch.utils.data.TensorDataset(train_input, train_label)    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
        
    validation_input = torch.from_numpy(validation_input)
    validation_label = torch.from_numpy(validation_label)
    validation_data = torch.utils.data.TensorDataset(validation_input, validation_label)    
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)


a = np.arange(18).reshape(2,3,3)

b = np.repeat(a, 8, axis = 1)

c = np.ones((10, 8 ))

d = np.concatenate((np.expand_dims(b, axis =2), np.expand_dims(c, axis =2)), axis = 2)

e = b[2:6:2]