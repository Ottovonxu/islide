import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import tarfile
import math
import time

# from config import config
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu
import numpy as np
from lazy_parser import MultiLabelDataset

from adam_base import Adam
from lsh_softmax import LSHSoftmax

np.random.seed(1234)
torch.manual_seed(1234)


DATASET = ['wiki10', 'amz13k','amz630k','deli','wiki300']
DATAPATH_TRAIN = {'wiki10': "/understand/learnLSH/data/wiki10_train.txt", 
                  'amz13k': "/understand/learnLSH/data/amazonCat_train.txt",
                  'amz630k': "/understand/learnLSH/data/amazon_shuf_train",
                  'deli':"/understand/learnLSH/data/deliciousLarge_shuf_train.txt",
                  'wiki300': "/understand/learnLSH/data/wikiLSHTC_shuf_train.txt"}
DATAPATH_TEST = {'wiki10': "/understand/learnLSH/data/wiki10_test.txt", 
                  'amz13k': "/understand/learnLSH/data/amazonCat_test.txt",
                  'amz630k': "/understand/learnLSH/data/amazon_shuf_test",
                  'deli':"/understand/learnLSH/data/deliciousLarge_shuf_test.txt",
                   'wiki300': "/understand/learnLSH/data/wikiLSHTC_shuf_test.txt"}


class Net(nn.Module):
    def __init__(self, IN, OUT):
        super(Net, self).__init__()
        self.IN = IN
        self.OUT = OUT
        self.fc = nn.Embedding(self.IN+1, 128, padding_idx=IN, sparse=True)

        self.bias = nn.Parameter(torch.Tensor(128))
        stdv = 1. / math.sqrt(self.IN)
        self.bias.data.uniform_(-stdv, stdv)

        self.smax = nn.Linear(128, self.OUT+1)

    def forward(self, x):
        emb = torch.sum(self.fc(x), dim=1)
        sizes = torch.sum(x != self.IN, dim=1, keepdim=True).float()
        mean_emb = emb / sizes
        query = F.relu(mean_emb + self.bias)
        return self.smax(query)

def train(args, model, device, loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    for batch_idx, (labels, data) in enumerate(loader):
        optimizer.zero_grad()
       # print('labels.size():',labels.size())
       # print('labels:',labels)
       # print('data.size():',data.size())

        batch_size = labels.size(0)
        targets = Variable(torch.zeros(batch_size, model.OUT+1))
        sizes = torch.sum(labels != -1, dim=1).int()
        for bdx in range(batch_size):
            num_labels = sizes[bdx].item()
            value = 1. / float(num_labels)
            for ldx in range(num_labels):
                targets[bdx, labels[bdx, ldx]] = value

        data, targets = data.to(device), targets.to(device)
        output_dist = F.log_softmax(model(data), dim=-1)
       # print('output_dist.size():',output_dist.size())
       # raise
        loss = F.kl_div(output_dist, targets, reduction='sum') / batch_size
        loss.backward()
        optimizer.step()

        interval = max(10, 1000//args.scale)
        if batch_idx % interval == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'
                  .format(epoch, batch_idx, len(loader), elapsed * 1000 / interval, loss.item()))
            sys.stdout.flush()
            start_time = time.time()

def evaluate(args, model, device, loader, MaxIter=None):
    model.eval()
    N = 0.
    correct = 0.
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (labels, data) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)
            output = model(data).cpu()
            values, indices = torch.max(output, dim=1)

            for bdx in range(batch_size):
                N += 1
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                if indices[bdx].item() in label_set:
                    correct+=1

            interval = max(10, 1000//args.scale)
          #  print('***interval,args.scale,batch_indx:****',interval,args.scale,batch_idx)
            if batch_idx % interval == 0:
                elapsed = time.time() - start_time
                print('| test {:5d}/{:5d} batches | ms/batch {:5.2f} | accuracy {:f} |'
                      .format(batch_idx, len(loader), elapsed * 1000 / interval, correct/N))
                sys.stdout.flush()
                start_time = time.time()
    print("Test Accuracy {:.4f}, total_correct {}".format(correct/N, correct))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type = str, default = "wiki10", choices = DATASET)
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default= 0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--scale', type=int, default=20, metavar='N',
                        help='batch size multiplier')
    parser.add_argument('--name', type=str, default="data/", metavar='N',
                        help='datapath')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = MultiLabelDataset(DATAPATH_TRAIN[args.dataset])
    #train_dataset = MultiLabelDataset('/understand/learnLSH/data/wiki10_train.txt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=True)

    # test_dataset = MultiLabelDataset('/understand/learnLSH/data/wiki10_test.txt')
    test_dataset = MultiLabelDataset(DATAPATH_TEST[args.dataset])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=False)

    print("Statistics:", train_dataset.N, train_dataset.D, train_dataset.L, train_dataset.max_D, train_dataset.max_L)
    model = Net(train_dataset.D, train_dataset.L).to(device)
    optimizer = Adam(model.parameters(), args.lr )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
       # if epoch % 10 == 0:
        #    evaluate(args, model, device, test_loader)
         #   print('-' * 89)
        evaluate(args, model, device, test_loader)
        print('-' * 89)


if __name__ == '__main__':
    main()
