import collections
import os
import sys
import math
import random
import numpy as np
import numpy.random
import scipy as sp
import scipy.stats

from clsh import pyLSH
import torch

class LSH:
    def __init__(self, func_, K_, L_, threads_=8):
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(self.K, self.L, threads_)

        self.sample_size = 0
        self.count = 0

    def stats(self):
        avg_size = self.sample_size // max(self.count, 1)
        print("Build", avg_size)
        self.sample_size = 0
        self.count = 0

    def remove_insert(self, item_id, old_item, new_fp):
        old_fp = self.func.hash(old_item).int().cpu().numpy()
        self.lsh_.remove(np.squeeze(old_fp), item_id)
        self.lsh_.insert(new_fp, item_id)

    def insert(self, item_id, item):
        fp = self.func.hash(item).int().cpu().numpy()
        self.lsh_.insert(np.squeeze(fp), item_id)

    def insert_fp(self, item_id, fp):
        self.lsh_.insert(np.squeeze(fp), item_id)

    def insert_multi(self, items, N):
        fp = self.func.hash(items).int().cpu().numpy()
        self.lsh_.insert_multi(fp, N)

    def query(self, item):
        fp = self.func.hash(item).int().cpu().numpy()
        return self.lsh_.query(np.squeeze(fp))

    def query_fp(self, fp):
        return self.lsh_.query(fp)

    def query_multi(self, items, N):
        fp = self.func.hash(items, transpose=True).int().cpu().numpy()
        return self.lsh_.query_multi(fp, N)

    def query_multi_mask(self, item, M, N):
        fp = self.func.hash(item).int().cpu().numpy()
        mask = torch.zeros(M, N, dtype=torch.float32)
        self.lsh_.query_multi_mask(fp, mask.numpy(), M, N)
        return mask.cuda()

    def accidental_match(self, labels, samples, N):
        self.lsh_.accidental_match(labels, samples, N)

    def multi_label(self, labels, samples):
        return self.lsh_.multi_label(labels, samples)

    def clear(self):
        self.lsh_.clear()









    # @cython.boundscheck(False)
    # def multi_label_nonunion(self, np.ndarray[long, ndim=2, mode="c"] labels, np.ndarray[long, ndim=2, mode="c"] samples, np.ndarray[long, ndim=3, mode="c"] sid_l):
    #     M = labels.shape[0]
    #     K = labels.shape[1]
    #     L = sid_l.shape[1]
    #     num_class = sid_l.shape[2]


    #     # remove accidental hits from samples
    #     # create label list
    #     # create label to index dictionary
    #     label_count =np.zeros(M)
    #     for idx in range(M): 
    #         for jdx in range(K): 
    #             l = labels[idx, jdx]
    #             if l == -1:
    #                 label_count[idx] = jdx
    #                 break
    #             if(jdx == K-1):
    #                 label_count[idx] = K
    #             samples[idx][l] = 0

        
    #     max_padding = max(np.sum(samples,axis=1) + label_count).astype("int")
    #     sample_L = np.zeros((M,L,max_padding))
    #     sample_list = np.zeros((M, max_padding)) + num_class

    #     for idx in range(M):
    #         content = np.concatenate( [ labels[idx][labels[idx]>=0], np.squeeze(np.argwhere( samples[idx] >0 ))])
    #         sample_list[idx,0: len(content)]  = content
    #         for l in range(L):
    #             l_content =  sid_l[idx,l,:][content]>0 
    #             sample_L[idx][l][:len(l_content)] = l_content
                


        
    #     label_count = label_count.astype("int")

    #     # create probability distribution
    #     result = np.zeros([M, max_padding], dtype=np.float32)
    #     for idx in range(M): 
    #         result[idx, 0:label_count[idx]] = 1/label_count[idx]
    #     return sample_list, result, sample_L