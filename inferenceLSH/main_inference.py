import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import tarfile
from torch.utils.tensorboard import SummaryWriter
import math
import time

import numpy as np
from lazy_parser import MultiLabelDataset

from adam_base import Adam
from lsh_softmax_inference import LSHSoftmax
from config import config
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# DATASET = ['wiki10', 'amz13k','amz630k','deli','wiki300']
# FREEZE_ACC = {'wiki10': 0.8, 'deli': 0.4, 'wiki300': 0.3}
# DATAPATH_TRAIN = {'wiki10': "/understand/learnLSH/data/wiki10_train.txt",
#                   'amz13k': "/understand/learnLSH/data/amazonCat_train.txt",
#                   'amz630k': "/understand/learnLSH/data/amazon_shuf_train",
#                   'deli':"/understand/learnLSH/data/deliciousLarge_shuf_train.txt",
#                   'wiki300': "/understand/learnLSH/data/wikiLSHTC_shuf_train.txt"}
# DATAPATH_TEST = {'wiki10': "/understand/learnLSH/data/wiki10_test.txt",
#                   'amz13k': "/understand/learnLSH/data/amazonCat_test.txt",
#                   'amz630k': "/understand/learnLSH/data/amazon_shuf_test",
#                   'deli':"/understand/learnLSH/data/deliciousLarge_shuf_test.txt",
#                    'wiki300': "/understand/learnLSH/data/wikiLSHTC_shuf_test.txt"}
DATASET = ['wiki10', 'amz13k','amz630k','deli','wiki300']
FREEZE_ACC = {'wiki10': 0.8, 'deli': 0.4, 'wiki300': 0.3}
DATAPATH_TRAIN = {'wiki10': "/home/bc20/NN/structured_matrix/wiki10_train.txt", 
                  'amz13k': "/home/bc20/NN/structured_matrix/amazonCat_train.txt",
                  'amz630k': "/home/bc20/NN/structured_matrix/amazon_shuf_train",
                  'deli':"/home/zl71/data/deliciousLarge_shuf_train.txt",
                  'wiki300': "/home/bc20/NN/data/wikiLSHTC_shuf_train.txt"}
DATAPATH_TEST = {'wiki10': "/home/bc20/NN/structured_matrix/wiki10_test.txt", 
                  'amz13k': "/home/bc20/NN/structured_matrix/amazonCat_test.txt",
                  'amz630k': "/home/bc20/NN/structured_matrix/amazon_shuf_test",
                  'deli':"/home/zl71/data/deliciousLarge_shuf_test.txt",
                   'wiki300': "/home/bc20/NN/data/wikiLSHTC_shuf_test.txt"}

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type = str, default = "deli", choices = DATASET)
parser.add_argument('--K', type = int, default = 18)
parser.add_argument('--L',type = int, default = 20)
parser.add_argument('--rebuild_freq',type = int, default = 30)  #wiki:30
parser.add_argument('--layer_dim',type = int, default = 128)
parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default= '0.005', metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--scale', type=int, default=20, metavar='N',help='batch size multiplier')
parser.add_argument('--name', type=str, default="data/", metavar='N',help='datapath')
parser.add_argument('--print_every', type = int, default = 500)
parser.add_argument('--test_every', type = int, default = 500)

args = parser.parse_args()

logfile = "./inference_log/{}/inference_K{}_L{}_r{}_b{}_lr{}_ryan.txt".format( args.dataset,args.K, args.L, args.rebuild_freq, args.batch_size, args.lr)
print("args: ",args, file = open(logfile, "a"))
print("args: ",args)

runfile = "./runs_correct/inference_{}_K{}_L{}_r{}_b{}_lr{}_ryan".format( args.dataset,args.K, args.L, args.rebuild_freq, args.batch_size, args.lr)
print("\nTensorboard file: ", runfile)
writer = SummaryWriter(runfile)

np.random.seed(123)
torch.manual_seed(123)

class Net(nn.Module):
    def __init__(self, IN, OUT):
        super(Net, self).__init__()
        self.IN = IN
        self.OUT = OUT
        self.fc = nn.Embedding(self.IN+1, args.layer_dim, padding_idx=IN, sparse=True)

        self.bias = nn.Parameter(torch.Tensor(args.layer_dim))
        stdv = 1. / math.sqrt(self.IN)
        self.bias.data.uniform_(-stdv, stdv)
        #print("bias",self.bias)
        self.smax = LSHSoftmax(N=OUT, D=args.layer_dim, K=args.K, L=args.L, freq=args.rebuild_freq)
        #self.smax = SampledSoftmax(ntokens=OUT, nsampled=OUT//6, nhid=128)

    def forward(self, x, y, freeze, slide = False, debug=False):
        raw_emb = torch.sum(self.fc(x), dim=1)
        emb = raw_emb / torch.norm(raw_emb, dim=1, keepdim=True)
        query = F.relu(emb + self.bias)
        return self.smax(query, y, freeze, slide, debug)

def train(args, model, device, loader, test_loader, optimizer, epoch, freeze):
    print("freeze", freeze)
    model.train()
    start_time = time.time()
    for batch_idx, (labels, data) in enumerate(loader):
        step = epoch * len(loader)+ batch_idx

        print_mode = batch_idx % args.print_every == args.print_every - 1

        optimizer.zero_grad()
        data = data.to(device)
        t1 = time.time()

        logits, new_targets, nsamples, avg_ip, avg_cos = model(data, labels, freeze, slide = False, debug = print_mode)

        output_dist = F.log_softmax(logits.view(-1, nsamples), dim=-1)
        batch_size = labels.size(0)
        loss = F.kl_div(output_dist, new_targets, reduction='sum') / batch_size
       # loss = F.binary_cross_entropy_with_logits(logits.view(-1, nsamples), new_targets, reduction='sum') / batch_size
        
        loss.backward()
        optimizer.step()

         #print train loss
        if print_mode:
            time_passed = time.time() - start_time
            print('===[%d, %5d] Network Train -> loss: %.3f' % (epoch , batch_idx + 1, loss.item()))
            print('===[%d, %5d] Time: %.5fs |'% (epoch, batch_idx + 1, time_passed))
            writer.add_scalar('task_loss/samples', nsamples/model.smax.params.weight.data.size()[0], step)
            writer.add_scalar('task_loss/avg_ip', avg_ip, step)
            writer.add_scalar('task_loss/avg_cos', avg_cos, step)
            writer.add_scalar('task_loss/train', loss.item(),step)

        if batch_idx % args.test_every  == args.test_every -1 :
            
            evaluate_slide(args, epoch, batch_idx, model, device, test_loader,training=True,k=5,slide=True)
            evaluate(args, epoch, batch_idx, model, device, test_loader,training=True,k=5,slide=False)

    print('-' * 89)

def evaluate_slide(args, epoch, batch_idx, model, device, loader, training, k=1,slide=True):
    freeze = False
    model.eval()
    N = 0.
    correct = 0.
    top1 = 0.
    stop=int(1000/args.test_batch_size)
    with torch.no_grad():
        for batch_idx, (labels, data) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)
            output, sample_size,id_list = model(data,labels,freeze, True)
            output=output.cpu()

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if id_list[indices[bdx, idx].item()] in label_set:
                        correct+=1.
                        if idx == 0:
                            top1 += 1

            if( batch_idx == stop and training ):
                break

    top1_acc = top1/N * k
    topk_acc = correct/N
    if training:
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
        print("[Slide: {}] Sample size: {}".format(slide, sample_size))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))
        print("[Slide: {}] Sample size: {}".format(slide, sample_size),file = open(logfile, "a"))
    else:
        print("End of Epoch: {}".format(epoch),file = open(logfile, "a"))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
        print("[Slide: {}] Sample size: {}".format(slide, sample_size))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))
        print("[Slide: {}] Sample size: {}".format(slide, sample_size),file = open(logfile, "a"))
    

    model.train()
    return top1_acc, topk_acc
        
  

def evaluate(args, epoch, iter, model, device, loader,training=False, k=1, slide=False):
    model.eval()
    freeze = False
    N = 0.
    correct = 0.
    top1 = 0.
    with torch.no_grad():
        for batch_idx, (labels, data) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)

            output = model.forward(data, labels, freeze = freeze, slide = slide, debug = False).cpu()

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if indices[bdx, idx].item() in label_set:
                        correct+=1.
                        if idx == 0:
                            top1 += 1

            if( batch_idx == 50 and training ):
                break

    top1_acc = top1/N * k
    topk_acc = correct/N
    if training:
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))
    else:
        print("End of Epoch: {}".format(epoch),file = open(logfile, "a"))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
        print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
        print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))
    model.train()
    return top1_acc, topk_acc

def main():

    train_dataset = MultiLabelDataset( DATAPATH_TRAIN[args.dataset])
    # train_dataset = MultiLabelDataset('/understand/learnLSH/data/deliciousLarge_shuf_train.txt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=True)

    test_dataset = MultiLabelDataset( DATAPATH_TEST[args.dataset])
    # test_dataset = MultiLabelDataset('/understand/learnLSH/data/deliciousLarge_shuf_test.txt')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=True)

    print("Statistics:", train_dataset.N, train_dataset.D, train_dataset.L, train_dataset.max_D, train_dataset.max_L)
    model = Net(train_dataset.D, train_dataset.L).to(device)

    # Embedding_weight = model.fc.weight.data.cpu().numpy()
    Embedding_bia = model.bias.data.cpu().numpy()
    # last_weight = model.smax.params.weight.data.cpu().numpy()
    # last_bias = model.smax.params.bias.data.cpu().numpy()
    # print("Embedding_weight",model.fc.weight.data.cpu().numpy()[0])
    # print("Embedding_bia",model.bias.data.cpu().numpy())
    # print("last_weight",model.smax.params.weight.data.cpu().numpy()[0])

    # np.save(".Embedding_weight",Embedding_weight )
    # np.save(".deli_Embedding_bia",Embedding_bia )
    # np.save(".last_weight",last_weight )
    # np.save(".last_bias",last_bias )
   # print('model.parameters:',model.parameters())
    optimizer = Adam(model.parameters(),lr = args.lr)

    freeze = False
    best_acc1 = 0.0
    best_acc5 = 0.0
    for epoch in range(0, args.epochs, 1):
        epoch_start_time = time.time()
        train(args, model, device, train_loader, test_loader, optimizer, epoch, freeze)
        _, _ = evaluate_slide(args, epoch, len(train_loader) +1 , model, device, test_loader, training=False, k=5, slide = True)
        top1_acc, top5_acc = evaluate(args, epoch, len(train_loader) +1 , model, device, test_loader, training=False, k=5, slide = False)

        is_best = (top1_acc > best_acc1) or (top5_acc > best_acc5)
        best_acc1 = max(  top1_acc,best_acc1)
        best_acc5 = max(  top5_acc,best_acc5)

        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)

        if(best_acc1> FREEZE_ACC[args.dataset] and freeze == False):
            freeze=True
            print("Set freeze weight")
            model.fc.weight.requires_grad = False
            model.bias.requires_grad = False
            print("Current Parameters")
            for p in model.parameters():
                if p.requires_grad:
                    print(p.name,p.size())

#    with open("output.txt", "a") as f:
#        print("Statistics:", train_dataset.N, train_dataset.D, train_dataset.L, train_dataset.max_D, train_dataset.max_L,file=f)  
#        model = Net(train_dataset.D, train_dataset.L).to(device)
#        optimizer = Adam(model.parameters())

#        for epoch in range(1, args.epochs+1, 1):
#            epoch_start_time = time.time()
#            train(args, model, device, train_loader, optprint('***interval,args.scale,batch_indx:****',interval,arg.scale,batch_indx)imizer, epoch,f)
#            print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)),file=f)
#            print('-' * 89,file=f)
#            evaluate(args, model, device, test_loader,f)
#            print('-' * 89,file=f)






if __name__ == '__main__':
    main()
