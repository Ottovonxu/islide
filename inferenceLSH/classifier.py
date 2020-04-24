import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import collections
from collections import Counter

class Network(nn.Module):
	def __init__(self, input_size, output_size, layer_size):
		super(Network, self).__init__()
		self.dense1 = nn.Linear(input_size, layer_size[0])
		self.dense1bn = nn.BatchNorm1d(layer_size[0])

		self.dense2 = nn.Linear(layer_size[0], output_size)

	def forward(self, x):
		x = F.relu(self.dense1bn(self.dense1(x)))
		x = self.dense2(x)
		return F.log_softmax(x,dim=1)
		#return F.log_softmax(x,dim=1)

class DatasetTrain(data.Dataset):
	def __init__(self, data, dim, num_class):
		self.data = data
		self.input_dim = dim
		self.num_class = num_class

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		return (self.data[index][:self.input_dim], self.data[index][self.input_dim:])

class DatasetTest(data.Dataset):
	def __init__(self, data, dim, num_class):
		self.x = data['x']

		y = data['y']
		length = max(map(len, y))
		self.y=np.array([xi+[-1]*(length-len(xi)) for xi in y])

		self.input_dim = dim
		self.num_class = num_class

	def __len__(self):
		return len(self.y)

	def __getitem__(self, index):
		return (self.x[index], self.y[index])
		#return (self.data[index][:self.input_dim], self.data[index][self.input_dim:])



def load_data(train_filepath, test_filepath,batch_size,feature_dim):
	full_data = np.load(train_filepath)
	test_data = np.load(test_filepath).item()

	#split train valid
	lengths = [int(len(full_data) * 0.8), len(full_data) - int(len(full_data) * 0.8) ]
	print("train valid data lengths", lengths)
	train_data, valid_data = data.random_split(full_data, lengths)

	# all_data = np.concatenate( (full_data, test_data), axis = 0)
	# np.random.shuffle(all_data)
	# train_data = all_data[:190000]
	# valid_data = all_data[190000:200000 ]
	# print(valid_data)
	# test_data = all_data[200000 : ]
	# print(test_data)

	train_Dataset = DatasetTrain(train_data, feature_dim, num_bins)
	train_dataloader = data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, num_workers=5)

	valid_Dataset = DatasetTrain(valid_data, feature_dim, num_bins)
	valid_dataloader = data.DataLoader(valid_Dataset, batch_size=len(valid_data), shuffle=True, num_workers=2)

	test_Dataset = DatasetTest(test_data,feature_dim,num_bins)
	test_dataloader = data.DataLoader(test_Dataset, batch_size=len(test_data['y']), shuffle=False, num_workers = 5)

	return train_dataloader, valid_dataloader, test_dataloader

if __name__== "__main__":

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:1" if use_cuda else "cpu")
	print("device", device)

	feature_dim = 128
	num_bins = 1000
	hidden_dim= [128]
	epoch_num = 20
	print_every = 3000
	batch_size = 32
	lr = 0.05

	train_filepath = './data/weight/amz680k/traindata1000_norm.npy'
	test_filepath = './data/weight/amz680k/querydata1000_norm.npy'

	#train_filepath = './data/weight/amz13k/traindata1000_nonorm.npy'
	#test_filepath = './data/weight/amz13k/querydata1000_nonorm.npy'

	train_dataloader, valid_dataloader, test_dataloader = load_data(train_filepath, test_filepath, batch_size, feature_dim)

	# print(next(iter(train_dataloader)))
	# x,y = next(iter(test_dataloader))
	# print(len(x))
	# print(y)
	# print( len(y))

	net = Network(feature_dim,num_bins,hidden_dim)
	loss_func = nn.NLLLoss() #loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
	net.to(device)
	print(net)

	train_loss_list = []
	valid_loss_list = []
	valid_acc_list=[]
	for epoch in range(epoch_num):

		running_loss = 0.0
		for i, batch in enumerate(train_dataloader):
			x, y = batch
			y =  y.reshape(-1)
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()

			output = net.forward(x.float())
			loss = loss_func(output, y.long())
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % print_every == print_every-1:  # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch , i + 1, running_loss / print_every))
				train_loss_list += [running_loss / print_every]
				running_loss = 0.0

		net.eval()
		with torch.no_grad():
			valid_loss = 0
			valid_acc = 0
			for x, y in valid_dataloader:
				y = y.reshape(-1)
				print(y)
				x, y = x.to(device), y.to(device)
				output = net.forward(x.float())
				loss = loss_func(output,y.long())
				output = torch.exp(output)
				top_p,top_c = output.topk(1,dim = 1)
				print(top_c.reshape(-1))
				equals = top_c.reshape(-1) == y
				valid_loss += loss.item()
				valid_acc += torch.mean(equals.type(torch.FloatTensor)).item()

			print('[%d] Validation loss: %.3f, Validation accuracy: %.3f\n' % (epoch, valid_loss/len(valid_dataloader),valid_acc / len(valid_dataloader)))
			valid_loss_list += [valid_loss/len(valid_dataloader)]
			valid_acc_list +=[valid_acc / len(valid_dataloader)]
		net.train()

	print("test model")
	test_loss = 0
	test_acc = 0
	net.eval()
	with torch.no_grad():
		for x, y in test_dataloader:
			#print(x)
			# print(y)
			print("average number of bin per sample: ",len(y.reshape(-1))/len(y))
			x, y = x.to(device), y.to(device)
			output = net.forward(x.float())

			#get class
			output = torch.exp(output)
			top_p,top_c = output.topk(1,dim = 1)

			top_c = top_c.reshape(-1).cpu().detach().numpy()
			print("predict:", top_c)
			acc = 0
			for i in range(len(y)):
				l = y[i,:].cpu().detach().numpy()
				# print(top_c[i])
				# print(l)
				if top_c[i] in l:
					acc+=1
			print("test accuracy", acc / len(top_c))


