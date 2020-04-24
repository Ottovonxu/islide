import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from simHash import SimHash
from lsh_new import LSH
import time
class LSHSoftmax(nn.Module):
    def __init__(self, N, D, K, L, freq):
        super(LSHSoftmax, self).__init__()
        self.D = D
        self.N = N
        self.K = K
        self.L = L

        # Rebuild Settings
        self.freq = freq
        self.count = 0
        self.sample_size = 0
        self.lsh = LSH(SimHash(D, K, L), K, L)
        
        self.params = nn.Linear(D, N)
        self.init_weights(self.params.weight, self.params.bias)

    def init_weights(self, weight, bias):
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

        self.lsh.insert_multi(self.params.weight.cuda().data, self.N)

    def build(self):
        #lsh.stats()
        self.lsh.clear()
        self.lsh.insert_multi(self.params.weight.cuda().data, self.N)

    def sampled(self, inputs, labels,freeze):
        # if self.lsh.count % self.freq == 0:
        #     self.build(self.lsh)

        # Query LSH Database
        t1 = time.time()
        N, D = inputs.size()
        sid = self.lsh.query_multi(inputs.data, N)

        sid_list, target_matrix = self.lsh.multi_label(labels.data.cpu().numpy(), sid)
        new_targets = Variable(torch.from_numpy(target_matrix)).cuda()

        #add y into bucket for inference
        if(freeze):
            # print("before insert:", len(self.lsh.query_multi(inputs.data, N)))
            query_hashcode = self.lsh.get_hashcode(inputs.data)
            for i, h in enumerate(query_hashcode):
                result = self.lsh.query_fp(h)
                temp_y = labels[i]
                temp_y = temp_y[ temp_y != -1]
                #print("temp_y", temp_y)
                for t in temp_y:
                    t = t.item()
                    if(result!=None):
                        if(t not in result):
                            #print(t)
                            self.lsh.insert_fp(t, h)
                            # print(self.lsh.query_fp(h))
                    else:
                        self.lsh.insert_fp(t, h)
            # print("after insert:", len(self.lsh.query_multi(inputs.data, N)))
            # print("sid_list", len(sid_list))


        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).cuda()
        sample_size = sample_ids.size(0)
        self.lsh.sample_size += sample_size
        self.lsh.count += 1
        
        # gather sample ids - weights and frequencies
        t1 = time.time()
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias[sample_ids]
        t1 = time.time()
        sample_logits = sample_weights.matmul(inputs.t()).t() + sample_bias

        #self.lsh.stats()

        return sample_logits, new_targets, sample_size


    def inference(self, inputs, labels,freeze):
        # if self.lsh.count % self.freq == 0:
        #     self.build(self.lsh)

        # Query LSH Database
        t1 = time.time()
        N, D = inputs.size()
        sid = self.lsh.query_multi(inputs.data, N)

        # sid_list, target_matrix = self.lsh.multi_label(labels.data.cpu().numpy(), sid)
        # new_targets = Variable(torch.from_numpy(target_matrix)).cuda()
        sid_list=list(sid)
        sample_ids = Variable(torch.from_numpy(np.asarray(sid_list, dtype=np.int64)), requires_grad=False).cuda()
        sample_size = sample_ids.size(0)
        self.lsh.sample_size += sample_size
        self.lsh.count += 1
        
        # gather sample ids - weights and frequencies
        t1 = time.time()
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = self.params.bias[sample_ids]
        t1 = time.time()
        sample_logits = sample_weights.matmul(inputs.t()).t() + sample_bias

        #self.lsh.stats()

        return sample_logits, sample_size,sid_list

    def forward(self, inputs, labels,freeze, slide):
        if self.training:
            return self.sampled(inputs, labels,freeze)
        else:
            if(slide):
                # logits,sample_sizes,sid_lists = 
                return self.inference(inputs, labels,freeze)
            else:
                logits = torch.matmul(inputs, self.params.weight.t()) + self.params.bias
                return logits
