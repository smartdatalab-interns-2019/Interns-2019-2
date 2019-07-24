import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data
from model import model
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from anomalyDetector import fit_norm_distribution_param
import os.path
import pickle
import create_data_for_deeplearning
from sklearn.model_selection import train_test_split
import numpy as np


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
        
        data_T_new = np.expand_dims(data_T, axis = 2)
        
        return data_T_new, data_T_new, Tag, timestamp_T, n_tag_0, n_tag_1, scale_norm, data_type
        
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

def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
    target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
    return data, target


def evaluate_1step_pred(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    with torch.no_grad():
        hidden = model.init_hidden(args.eval_batch_size)
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):

            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            outSeq, hidden = model.forward(inputSeq, hidden)

            loss = criterion(outSeq.contiguous().view(args.batch_size,-1), targetSeq.contiguous().view(args.batch_size,-1))
            hidden = model.repackage_hidden(hidden)
            total_loss+= loss.item()

    return total_loss / nbatch

def denormalized_data(normalized_dataset, scale_norm, DATA_ORGANIZATION, data_type1):
    
    #normalized_dataset = gen_dataset[:figNumber].cpu().numpy()
    
    dict = ['correlation_coeff_max', 'correlation_coeff_min', 'temperature_max', 'temperature_min', 'humidity_max', 'humidity_min']
    
    if DATA_ORGANIZATION == 'full_combination':
        
        recovered_dataset = np.zeros_like(normalized_dataset)
        
        recovered_dataset[:, :, 2:] = normalized_dataset[:, :, 2:] * (scale_norm[dict[0]] - scale_norm[dict[1]])\
        + scale_norm[dict[1]]
        for i in range(2):
            recovered_dataset[:, :, i] = normalized_dataset[:, :, i] * (scale_norm[dict[2*(i+1)]] - scale_norm[dict[2*(i+1)+1]])\
            + scale_norm[dict[2*(i+1)+1]]
            
    elif DATA_ORGANIZATION == 'partial_combination':
        recovered_dataset = np.zeros_like(normalized_dataset)        
        for i in range(3):
            recovered_dataset[:, :, i] = normalized_dataset[:, :, i] * (scale_norm[dict[2*i]] - scale_norm[dict[2*i+1]])\
            + scale_norm[dict[2*i+1]]
        
    else:
        recovered_dataset = []
        for i in range(np.shape(normalized_dataset)[0]):
            
            temp_data_type = data_type1[i]
            temp_recovered_data = normalized_dataset[i] * (scale_norm[dict[2 * temp_data_type]] - scale_norm[dict[2 * temp_data_type + 1]]) \
            + scale_norm[dict[2 * temp_data_type + 1]]
            recovered_dataset.append(temp_recovered_data)
        
        recovered_dataset = np.array(recovered_dataset)        
    
    return recovered_dataset

def generate_output(args, epoch, model, gen_dataset, scale_norm, data_organization, disp_uncertainty=True, figNumber = 30, startPoint = 50, endPoint = 400):
    
    if args.save_fig:
        # Turn on evaluation mode which disables dropout.
        model.eval()
        
        outSeq = []
        # upperlim95 = []
        # lowerlim95 = []
        
        for n in range(figNumber):
            tempOutSeq = []
            hidden = model.init_hidden(1)
            with torch.no_grad():
                for i in range(endPoint):
                    if i>=startPoint:
                        # if disp_uncertainty and epoch > 40:
                        #     outs = []
                        #     model.train()
                        #     for i in range(20):
                        #         out_, hidden_ = model.forward(out+0.01*Variable(torch.randn(out.size())).cuda(),hidden,noise=True)
                        #         outs.append(out_)
                        #     model.eval()
                        #     outs = torch.cat(outs,dim=0)
                        #     out_mean = torch.mean(outs,dim=0) # [bsz * feature_dim]
                        #     out_std = torch.std(outs,dim=0) # [bsz * feature_dim]
                        #     upperlim95.append(out_mean + 2.58*out_std/np.sqrt(20))
                        #     lowerlim95.append(out_mean - 2.58*out_std/np.sqrt(20))
    
                        out, hidden = model.forward(out, hidden)
    
                        #print(out_mean,out)
    
                    else:
                        out, hidden = model.forward(gen_dataset[n][i].unsqueeze(0).unsqueeze(0).float(), hidden)
                    tempOutSeq.append(out)
                    
                tempOutSeq = torch.cat(tempOutSeq, dim=1)
            outSeq.append(tempOutSeq)

        outSeq = torch.cat(outSeq, dim=0) # [seqLength * feature_dim]

        target = denormalized_data(gen_dataset[:figNumber].cpu().numpy(), scale_norm, DATA_ORGANIZATION, data_type)

        outSeq = denormalized_data(outSeq.cpu().numpy(), scale_norm, DATA_ORGANIZATION, data_type)
  
        # if epoch>40:
        #     upperlim95 = torch.cat(upperlim95, dim=0)
        #     lowerlim95 = torch.cat(lowerlim95, dim=0)
        #     upperlim95 = preprocess_data.reconstruct(upperlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)
        #     lowerlim95 = preprocess_data.reconstruct(lowerlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)

        if data_organization == 'partial_combination':
            for i in range(target.shape[0]):
                fig = plt.figure(figsize=(15,5))
                plt.axis('off')
                plt.grid(b=None)
                plt.title('Time-series Prediction on ' + args.data + ' Dataset', y = 1.05, fontsize=18, fontweight='bold')
                for j in range(3):     
                    ax = fig.add_subplot(3, 1, j+1)
                    ax.plot(target[i,:,j], label='Target'+str(i),
                             color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                    ax.plot(range(startPoint), outSeq[i,:startPoint,j], label='1-step predictions for target'+str(i),
                             color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    # if epoch>40:
                    #     plt.plot(range(startPoint, endPoint), upperlim95[:,i].numpy(), label='upperlim'+str(i),
                    #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    #     plt.plot(range(startPoint, endPoint), lowerlim95[:,i].numpy(), label='lowerlim'+str(i),
                    #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    ax.plot(range(startPoint, endPoint), outSeq[i,startPoint:,j], label='Recursive predictions for target'+str(i),
                             color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    
                    # plt.xlim([startPoint-500, endPoint])
                    plt.xlabel('Index',fontsize=15)
                    plt.ylabel('Value',fontsize=15)
                    plt.legend()
                plt.subplots_adjust(wspace = 0.2, hspace = 0.3)
                # plt.tight_layout()
                # plt.text(startPoint-500+10, target.min(), 'Epoch: '+str(epoch),fontsize=15)                               
                save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_prediction')
                save_dir.mkdir(parents=True,exist_ok=True)
                plt.savefig(save_dir.joinpath('fig_epoch'+str(epoch)+'_'+str(i+1)).with_suffix('.png'))
                plt.show()
                plt.close()
        elif data_organization == 'full_combination':
            for i in range(target.shape[0]): 
                fig = plt.figure(figsize=(15,5))
                plt.axis('off')
                plt.grid(b=None)
                plt.title('Time-series Prediction on ' + args.data + ' Dataset', y = 1.05, fontsize=18, fontweight='bold')
                for j in range(4):
                    
                    ax = fig.add_subplot(2, 2, j+1)
                    ax.plot(target[i,:,j], label='Target'+str(i),
                             color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
                    ax.plot(range(startPoint), outSeq[i,:startPoint,j], label='1-step predictions for target'+str(i),
                             color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    # if epoch>40:
                    #     plt.plot(range(startPoint, endPoint), upperlim95[:,i].numpy(), label='upperlim'+str(i),
                    #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    #     plt.plot(range(startPoint, endPoint), lowerlim95[:,i].numpy(), label='lowerlim'+str(i),
                    #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
                    ax.plot(range(startPoint, endPoint), outSeq[i,startPoint:,j], label='Recursive predictions for target'+str(i),
                             color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)    
                    
                    # plt.xlim([startPoint-500, endPoint])
                    plt.xlabel('Index',fontsize=15)
                    plt.ylabel('Value',fontsize=15)
                    plt.legend()
                    
                plt.subplots_adjust(wspace = 0.2, hspace = 0.3)
                # plt.tight_layout()
                # plt.text(startPoint-500+10, target.min(), 'Epoch: '+str(epoch),fontsize=15)                
                # plt.show()
                save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_prediction')
                save_dir.mkdir(parents=True,exist_ok=True)
                plt.savefig(save_dir.joinpath('fig_epoch'+str(epoch)+'_'+str(i+1)).with_suffix('.png'))
                plt.show()
                plt.close()
        return outSeq

    else:
        pass


def train(args, model, train_dataset, epoch):

    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,train_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.repackage_hidden(hidden)
            hidden_ = model.repackage_hidden(hidden)
            optimizer.zero_grad()

            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals=[]
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1.contiguous().view(args.batch_size,-1), targetSeq.contiguous().view(args.batch_size,-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.contiguous().view(args.batch_size, -1), targetSeq.contiguous().view(args.batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.contiguous().view(args.batch_size,-1), hids2.contiguous().view(args.batch_size,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3
            
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                      'loss {:5.5f} '.format(
                    epoch, batch, len(train_dataset) // args.bptt,
                                  elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()
            

def evaluate(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(args.eval_batch_size)
        nbatch = 1
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]
            hidden_ = model.repackage_hidden(hidden)
            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals=[]
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1.contiguous().view(args.batch_size,-1), targetSeq.contiguous().view(args.batch_size,-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.contiguous().view(args.batch_size, -1), targetSeq.contiguous().view(args.batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.contiguous().view(args.batch_size,-1), hids2.contiguous().view(args.batch_size,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3

            total_loss += loss.item()

    return total_loss / (nbatch+1)

###############################################################################
# hyperparameters
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset', conflict_handler = 'resolve')
parser.add_argument('--data', type=str, default='plate_vibration_signal_full_combination',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='vibration_signal_process_model.pkl',
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
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true', default = True,
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p', default = True,
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

DATA_ORGANIZATION = 'full_combination'
N_FILE_TYPE = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args.pretrained = False
args.resume = False
args.save_fig = True
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

    filename_preprocess_data = 'D:/Research/Kang/Data/plate_ultrasonic_dataset_197_process_' + DATA_MODE + '.pickle'
        
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
            
        train_input, validation_input, train_label, validation_label,  data_type_train, data_type_validation = \
        train_test_split(data_T[no_mass_index,:], label_T[no_mass_index,:], data_type[no_mass_index], test_size = 0.2)        
    
    validation_input, test_input, validation_label, test_label, data_type_validation, data_type_test = \
    train_test_split(validation_input, validation_label, data_type_validation, test_size = 0.2)    
    
    train_input = torch.from_numpy(train_input)
    train_label = torch.from_numpy(train_label)       
    train_data = torch.utils.data.TensorDataset(train_input, train_label)    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
        
    validation_input = torch.from_numpy(validation_input)
    validation_label = torch.from_numpy(validation_label)
    validation_data = torch.utils.data.TensorDataset(validation_input, validation_label)    
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = args.batch_size, shuffle = True)

    test_input = torch.from_numpy(test_input)
    test_label = torch.from_numpy(test_label)
    test_data = torch.utils.data.TensorDataset(test_input, test_label)    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True) 

###############################################################################
# Build the model
###############################################################################
feature_dim = train_input.size(2)
model = model.RNNPredictor(rnn_type = args.model,
                           enc_inp_size = feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size = feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           tie_weights= args.tied,
                           res_connection=args.res_connection).to(torch.device(args.device))
optimizer = optim.Adam(model.parameters(), lr= args.lr,weight_decay=args.weight_decay)
criterion = nn.MSELoss()

###############################################################################
# Training code
###############################################################################
#args.resume = True
#args.pretrained =  True

# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = torch.load(Path('D:/Research/Kang/Research/Anomaly Detection Based on RNN/save/plate_vibration_signal_partial_combination/model_best/vibration_signal_process_model.pth'))
    args, start_epoch, best_val_loss = model.load_checkpoint(args, checkpoint, feature_dim)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = 0
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

if not args.pretrained:
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, args.epochs+1):

            epoch_start_time = time.time()
            for i, batch in list(enumerate(train_loader))[:-1]:
                
                train_dataset = batch[0].transpose_(0, 1).to(device)
            
                train(args, model, train_dataset.float(), epoch)
                
                if i % 50 == 0:
                    print("\n{:4d} batch files have been trained\n".format(i))
            
            val_loss = 0
            for i, batch in list(enumerate(validation_loader))[:-1]:
            
                test_dataset = batch[0].transpose_(0, 1).to(device)
                cur_val_loss = evaluate(args, model, test_dataset.float())
                val_loss = val_loss + cur_val_loss
            
            val_loss = val_loss / i
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time), val_loss))
            print('-' * 89)

            # generate_output(args,epoch,model,gen_dataset,startPoint=1500)

            if epoch%args.save_interval==0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss > best_val_loss
                best_val_loss = max(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args':args
                                    }
                model.save_checkpoint(model_dictionary, is_best)
            
            if epoch % 50 == 0:
                result_demonstration = generate_output(args, epoch, model, test_input.to(device), \
                                                       scale_norm, DATA_ORGANIZATION, disp_uncertainty=True, \
                                                       figNumber = 30, startPoint = 50, endPoint = 400)
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

args.save_fig = True