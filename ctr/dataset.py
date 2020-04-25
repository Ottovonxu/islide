import torch
from torch.utils import data
import numpy as np
from sklearn.utils import murmurhash3_32
import pdb
from densified_minhash import densified_minhash as dnmh
import pandas as pd
import random
import seaborn as sns

import matplotlib.pyplot as plt

def Hfunction(m, seed=None):
    #if seed is None:
    #    seed = np.random.randint(0,100000)
    return lambda x : murmurhash3_32(key=x, seed=seed, positive=True) % m

def Gfunction(seed=None):
    #if seed is None:
    #    seed = np.random.randint(0,100000)
    return lambda x : np.sign(murmurhash3_32(key=x, seed=seed))

'''
    This can be used to do a feature hashing to the dimension 
    given
'''
class FHDataset(data.Dataset):
    def __init__(self, X_file, y_file, dimension):
        super(FHDataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        with open(y_file, 'r+') as yfile:
            self.y = yfile.readlines()
        self.length = len(self.X)
        self.dim = dimension
        self.h = Hfunction(dimension, seed=7989)
        self.g = Gfunction(seed=101)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        nonzeros = self.X[index].replace("\n", "").split(",")
        label = int(self.y[index].replace("\n", ""))
        for idx in nonzeros:
            hidx = self.h(int(idx))
            data_point[hidx] = data_point[hidx] + self.g(int(idx))
        return data_point, label

'''
    This can be used to do a bbit encoding followed by FH of the minhash data
    to the specified dimension.
    This opens the minhash data.
'''
class MHBbitDataset(data.Dataset):
    def __init__(self, X_file, y_file, dimension, bbits):
        super(MHDataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        with open(y_file, 'r+') as yfile:
            self.y = yfile.readlines()
        
        k = len(self.X[0].replace("\n", "").split(","))
        
        self.bbits = bbits
        self.size_perhash = 2**bbits
        
        self.length = len(self.X)
        mask = ~0
        mask = mask << bbits
        mask = ~mask
        self.mask = mask
        print("MHDataset:", self.bbits, "reduced_dimension", dimension)
        self.h = Hfunction(dimension, seed=7989)
        self.g = Gfunction(seed=101)
        self.dim = dimension


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        mhs = self.X[index].replace("\n", "").split(",")
        label = int(self.y[index].replace("\n", ""))
        for key_i in range(0, len(mhs)):
            mh = mhs[key_i]
            bitmh = int(mh) & self.mask
            idx = key_i * self.size_perhash + bitmh
            hidx = self.h(int(idx))
            data_point[hidx] = data_point[hidx] + self.g(int(idx))
        return data_point, label

''' Simple dataset which will create a one hot vector of the
    given co-ordinates
'''

class Dataset(data.Dataset):
    def __init__(self, X_file, y_file, dimension):
        super(Dataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        with open(y_file, 'r+') as yfile:
            self.y = yfile.readlines()
        self.length = len(self.X)
        self.dim = dimension

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        nonzeros = self.X[index].replace("\n", "").split(",")
        label = int(self.y[index].replace("\n", ""))
        for idx in nonzeros:
            data_point[int(idx)] = 1.
        return data_point, label


''' 
    Given a raw dataset, it will compute Min hashes and return
    the embedding
    data style : single file
    <label> <data co-ordinates>
'''

class SingleFileDirectMHDataset(data.Dataset):
    def __init__(self, X_file, dimension, K, pairwise, hashFullDMH=False):
        super(SingleFileDirectMHDataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = dimension
        self.DMH = dnmh.Densified_MinHash(K, dimension, seed=1012, num_seed=2024, hashFull=hashFullDMH)
        self.pairwise = pairwise
        print("SingleFileDirectMHDataset", X_file, "Pairwise: ", self.pairwise)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].replace("\n", "").split(" ")
        #data = [int(x) for x in data]
        label = int(data[0])
        if label==-1:
            label=0
        xdata = data[1:]
        xdata=[item.split(":")[0] for item in xdata]

        if self.pairwise:
          pdata = []
          for i in range(0, len(xdata)):
            for j in range(i+1, len(xdata)):
              pdata.append(xdata[i]+"."+xdata[j])
          xdata = pdata

        mhdata = self.DMH.get_hashed(xdata)
        for idx in mhdata:
            data_point[int(idx)] = 1.
        return data_point, label



class SingleFileDirectMHDatasetFast(data.Dataset):
    def __init__(self, X_file, dimension, K, pairwise, hashFullDMH, method):
        super(SingleFileDirectMHDatasetFast, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = dimension
        self.DMH = dnmh.Densified_MinHash(K, dimension, seed=1012, num_seed=2024, hashFull=hashFullDMH)
        self.pairwise = pairwise
        self.method = method
        print("SingleFileDirectMHDatasetFAST Method:",method, "Pairwise:", pairwise,"file:", X_file)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].replace("\n", "").split(" ")
        #data = [int(x) for x in data]
        label = int(data[0])
        if label==-1:
            label=0
        xdata = data[1:]
        xdata=[item.split(":")[0] for item in xdata]

        if self.pairwise:
          pdata = []
          for i in range(0, len(xdata)):
            for j in range(i+1, len(xdata)):
              pdata.append(xdata[i]+"."+xdata[j])
          xdata = pdata
        if self.method == "univ":
            data_point = self.DMH.get_hashed_4universal(xdata)
        else:
            data_point = self.DMH.get_hashed_faster(xdata)
        return data_point, label

'''
    This can be used to do a feature hashing to the dimension 
    given
'''
class SingleFileDirectFHDataset(data.Dataset):
    def __init__(self, X_file, dimension, pairwise):
        super(SingleFileDirectFHDataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = dimension
        self.h = Hfunction(dimension, seed=7989)
        self.g = Gfunction(seed=101)
        self.pairwise = pairwise
        print("SingleFileDirectFhDataset", X_file, "pairwise:", self.pairwise)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].replace("\n", "").split(" ")
        #data = [int(x) for x in data]
        label = int(data[0])
        if label==-1:
            label=0
        xdata = data[1:]
        xdata=[item.split(":")[0] for item in xdata]
        if self.pairwise:
          pdata = []
          for i in range(0, len(xdata)):
            for j in range(i+1, len(xdata)):
              pdata.append(xdata[i]+"."+xdata[j])
          xdata = pdata
        for idx in xdata:
            hidx = self.h(idx)
            data_point[hidx] = data_point[hidx] + self.g(idx)
        return data_point, label





class SimpleDataset(data.Dataset):
    def __init__(self, X_file, dimension):
        super(SimpleDataset, self).__init__()
        self.d = pd.read_csv(X_file, sep=" ", header=None)
        len1 = len(self.d)
        self.d.dropna(inplace=True)
        len2 = len(self.d)
        self.d = self.d.astype(np.int32)
        self.d.columns = ['label'] + ['C'+str(i) for i in range(len(self.d.columns) - 1)]
        self.labels = self.d['label'].values
        self.d.drop('label', inplace=True, axis=1)
        self.length = len(self.d)
        self.dim = dimension

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        xdata = self.d.iloc[index]
        label = self.labels[index] 
        data_point[xdata] = 1
        return data_point, label

def get_lrdataset(tfile, dimension):
  return SimpleDataset(tfile, dimension)

def get_dataset(tfile, dimension, MHTrain, K, pairwise, hashFullDMH, method):
    if MHTrain:
        if method!="orig":
          return SingleFileDirectMHDatasetFast(tfile, dimension, K, pairwise, hashFullDMH,method)
        else:
          return SingleFileDirectMHDataset(tfile, dimension, K, pairwise, hashFullDMH)
    else:
        return PureDataset(tfile,dimension)

    # else:
    #     return SingleFileDirectFHDataset(tfile, dimension, pairwise)
class PureDataset(data.Dataset):
    def __init__(self, X_file, dimension):
        super(PureDataset, self).__init__()
        with open(X_file, 'r+') as xfile:
            self.X = xfile.readlines()
        self.length = len(self.X)
        self.dim = dimension
        
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data_point = np.zeros(self.dim)
        data = self.X[index].replace("\n", "").split(" ")
        #data = [int(x) for x in data]
        label = int(data[0])
        if label==-1:
            label=0
        xdata = data[1:]
        xdata=[int(item.split(":")[0]) for item in xdata]
        # print(xdata)
        data_point[xdata]=1
        return data_point, label
if __name__ == '__main__':
    dataset = PureDataset(X_file="./url/data/train.txt", dimension=3231961)
    print(dataset[1])

    #dataset = FHDataset(X_file="./rcv1/data/train_X.txt", y_file="./rcv1/data/train_y.txt", dimension=10000)
    #x = dataset[1]

    # datasetx = SingleFileDirectMHDataset(X_file='./rcv1/data/train.txt', dimension=100, K=10)
    # x = datasetx[1]

    # datasety = SingleFileDirectFHDataset(X_file='./url/data/test.txt', dimension=64, pairwise=False)

    # print(datasety[1][1])

    # id_list=[i for i in range(len(datasety))]
    # random.shuffle(id_list)

    # id_list=id_list[:10]
    # label=np.random.random(size=(10,))
    # label=2*label-1
    # for i in range(5):
    #     data=datasety[id_list[i]][0]
    #     matrix1=np.expand_dims(data,1)
    #     matrix2=np.expand_dims(data,0)
    #     if i==0:
    #         matrix=np.dot(matrix1,matrix2)*label[i]
    #     else:
    #         matrix+=np.dot(matrix1,matrix2)*label[i]
    # # print(matrix)
    # print(np.linalg.matrix_rank(matrix))
    # print(np.linalg.norm(matrix))

    # eigs=np.linalg.eigh(matrix)[0]
    # plt.figure()
    # plt.plot(np.arange(len(eigs)),np.sort(np.abs(eigs)))
    # plt.savefig("eig_FH.jpg")
    # # print(np.linalg.eigh(matrix)[0])

    # matrix=np.abs(matrix)
    # min_val=np.min(matrix)
    # max_val=np.max(matrix)
    # sns.set()
    # # plt.imshow(matrix, cmap='hot', interpolation='none')
    # # plt.figure()
    # # ax = sns.heatmap(matrix,cmap="gray",vmin=min_val, vmax=max_val)
    # # plt.savefig("heat_FH.jpg")


    # datasetx = SingleFileDirectMHDatasetFast(X_file='./criteo/data/train_small_ub.txt', dimension=64, K=32 ,pairwise=False,hashFullDMH=False,method="univ")


    # for i in range(5):
    #     data=datasetx[id_list[i]][0]
    #     matrix1=np.expand_dims(data,1)
    #     matrix2=np.expand_dims(data,0)
    #     if i==0:
    #         matrix=np.dot(matrix1,matrix2)*label[i]
    #     else:
    #         matrix+=np.dot(matrix1,matrix2)*label[i]
    # # print(matrix)
    # print(np.linalg.matrix_rank(matrix))
    # print(np.linalg.norm(matrix))
    # print(np.linalg.eigh(matrix)[0])

    # eigs=np.linalg.eigh(matrix)[0]
    # plt.figure()
    # plt.plot(np.arange(len(eigs)),np.sort(np.abs(eigs)))
    # plt.savefig("eig_MH.jpg")

    # matrix=np.abs(matrix)

    # sns.set()

    # plt.imshow(matrix, cmap='hot', interpolation='none')
    # plt.figure()
    # ax = sns.heatmap(matrix,cmap="gray",vmin=min_val, vmax=max_val)
    # plt.savefig("heat_MH.jpg")






    # nonzero_count=np.zeros((len(datasety),))
    # for i in range(len(datasety)):
    #     data=datasety[i][0]
    #     nonzero_count[i]=np.count_nonzero(data)
    # print(np.var(nonzero_count))

    # pdb.set_trace()
