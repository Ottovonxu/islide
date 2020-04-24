import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from simHash import SimHash
from lsh_new import LSH
from config import config
import time
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_packed_sequence



# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSHSampledLayer(nn.Module):
    def __init__(self, layer_size, K, L,num_class):
        super(LSHSampledLayer, self).__init__()
        self.D = layer_size
        self.K = K 
        self.L = L
        self.num_class = num_class
        
        self.store_query = True
        #last layer
        self.params = nn.Linear(layer_size, num_class + 1)
        self.init_weights(self.params.weight, self.params.bias)

        #construct lsh using triplet weight
        self.lsh = None
        self.buildLSH()

        self.count = 0
        self.sample_size = 0

    def buildLSH(self):
        #print("in lshlayer rebuilt lshtable")
        if(self.lsh == None):
            self.lsh = LSH( SimHash(self.D, self.K, self.L), self.K, self.L )
        
        else:
            self.lsh.resetLSH( SimHash(self.D, self.K, self.L ) )
        

        self.lsh.insert_multi(self.params.weight.to(device).data, self.num_class )

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        weight.data[:,-1].zero_()
        bias.data.fill_(0)
    
    def train_forward(self, x, y):

        N, D = x.size()
        sid = self.lsh.query_multi(x.data, N)
        sid_list, target_matrix = self.lsh.multi_label(y.data.cpu().numpy(), sid)
        new_targets = Variable(torch.from_numpy(target_matrix)).to(device)
        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).to(device)
        sample_size = sample_ids.size(0)
    
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias[sample_ids]
        sample_product = sample_weights.matmul(x.t()).t()
        sample_logits = sample_product + sample_bias

        self.lsh.sample_size += sample_size
        self.lsh.count += 1

        #self.lsh.stats()
        return sample_logits, new_targets, sample_size   

    def forward(self, x, y,freeze_flag):
        if(freeze_flag != True):
            if self.training:
                return self.train_forward(x, y)
            else:
                return torch.matmul(x,self.params.weight.t()) + self.params.bias, 0,0 
        else:

            return self.train_forward(x, y)

class Net(nn.Module):
    def __init__(self, input_size, output_size, layer_size, K, L):
        super(Net, self).__init__()
        stdv = 1. / math.sqrt(input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.fc = nn.Embedding(self.input_size+1, 128, padding_idx=input_size, sparse=True)
        self.bias  = nn.Parameter(torch.Tensor(layer_size))
        self.bias.data.uniform_(-stdv, stdv)
        self.lshLayer = LSHSampledLayer(layer_size, K, L, output_size)

    def forward(self, x, y, unfreeze_flag):
        raw_emb = torch.sum(self.fc(x), dim=1)
        emb = raw_emb / torch.norm(raw_emb, dim=1, keepdim=True)
        query = F.relu(emb + self.bias)
        return self.lshLayer.forward(query, y,unfreeze_flag) 


class MultiLabelDataset(Dataset):
    def __init__(self, filename):
        self.build(filename)

    def build(self, filename):
        with open(filename) as f:

            metadata = f.readline().split()
            self.N = int(metadata[0])
            self.D = int(metadata[1])
            self.L = int(metadata[2])
            
            self.max_L = 0
            self.max_D = 0
            
            
            self.data = list()
            for idx in range(self.N):
                items = f.readline().split()
                labels = [int(x) for x in items[0].split(",")]
                self.max_L = max(self.max_L, len(labels))
                
                ids = list()
                # fvs = list()
                for fdx in range(1, len(items), 1):
                    fid, fv = items[fdx].split(":")
                    ids.append( int(fid) )
                    # fvs.append( float(fv) )
                self.max_D = max(self.max_D, len(ids))
                self.data.append( [torch.from_numpy(np.asarray(x)) for x in [labels, ids]] )

                # if idx % 100000 == 0:
                #     print(idx)

    def pad(self, item, width, value):
        result = torch.zeros(width).long()
        result.fill_(value)
        result[:len(item)] = item
        return result

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        labels, idxs = self.data[idx]
        return self.pad(labels, self.max_L, -1), self.pad(idxs, self.max_D, self.D)

