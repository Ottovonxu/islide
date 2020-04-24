import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

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
        print(num_feat)
        features, labels = load_svmlight_file(f,n_features = num_feat, multilabel=True)

    return features, labels

def ReLU(x):
    return x * (x > 0)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# feature_dim = 782585
# hidden1_dim = 128 #128
# hidden2_dim = 205443 #205443


feature_dim = 101938
hidden1_dim = 128 #128
hidden2_dim = 30938
features, labels  = read_data('/home/bc20/NN/structured_matrix/wiki10_test.txt',header=True)
data = np.load('/home/zl71/learnLSH/data/nn-weight/weight_wiki10k.npz')



#features, labels  = read_data('/home/bc20/NN/data/deliciousLarge_shuf_test.txt')

# data = np.load("/home/zx22/slide/branch/HashingDeepLearning/saved_weights/delicious2.npz")
# w0 = data['w_layer_0']
# b0 = data['b_layer_0']
# w1 = data['w_layer_1']
# b1 = data['b_layer_1']

#data = np.load("/home/zx22/slide/branch/HashingDeepLearning/python_examples/deli/toEmma/weight_deli.npz")
w0 = data['W1']
b0 = data['b1']
w1 = data['W2']
b1 = data['b2']

features = features.toarray()
# pad = np.zeros( shape = (6616,1))
# features = np.concatenate( (features,pad), axis = 1)
print("feautre shape", features.shape)
print("w0 shape",w0.shape)
print("w1 shape", w1.shape)
print("b0 shape",b0.shape)
print("b1 shape", b1.shape)


# output0 = ReLU(np.matmul(features, np.transpose(w0)) + b0)
# print("output0 shape", output0.shape)
# print(output0)
# logits =  ReLU(np.matmul(output0, np.transpose(w1)) + b1)
# print("logits shape", logits.shape)
# result = np.argmax(logits, axis = 1)

output0 = ReLU(np.matmul(features, w0) + b0)
print("output0 shape", output0.shape)
print(output0)
logits =  np.matmul(output0, w1) + b1
print("logits shape", logits.shape)
result = np.argmax(logits, axis = 1)
print(len(result))
acc = 0
index = 0
for l in labels:
    #print("predict", result[0])
    if result[index] in list(l):
        acc+=1
    index +=1
    #print("in")
print(acc / len(result))




