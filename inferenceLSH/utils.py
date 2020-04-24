import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn import preprocessing
import os
import sys


def get_query_weight(datafile, weightfile):

    # read data file
    #features, labels, num_feat, num_labels = read_data(datafile, header=True)
    features = read_data(datafile, header=True)
    features = features.toarray()
    w0, b0, w1, b1 = read_weight(weightfile)

    query =np.matmul(features, w0) + b0

    #normalize 
    # w1 = preprocessing.normalize(w1, norm = 'l1', axis = 1)
    # query = preprocessing.normalize(query, norm = 'l1', axis = 1)

    # print("feautre shape", features.shape)
    # print("w0 shape",w0.shape)
    # print("w1 shape", w1.shape)
    # print("query", query.shape)

    return query, w1

def read_data(filename, header=False, dtype='float32', zero_based=True):
    with open(filename, 'rb') as f:
        _l_shape = None
        if header:
            line = f.readline().decode('utf-8').rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(line[0]), int(line[1]), int(line[2])
            _l_shape = (num_samples, num_labels)
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(f,n_features = num_feat,  multilabel=True)
    return features
    # return features.toarray(), labels, num_feat, num_labels


def read_weight(filename):
    data = np.load(filename)
    w0 = data['W1']
    b0 = data['b1']
    w1 = data['W2']
    b1 = data['b2']
    w1 = np.transpose(w1)

    return w0, b0, w1, b1

