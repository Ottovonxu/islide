import os
import sys
import numpy as np
import argparse
from datetime import datetime
import time

#torch related 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import torch.nn.init as init

from config import config
from network import MultiLabelDataset, Net
from triplet_network import Network, TripletNet
from adam_base import Adam

# from types import SimpleNamespace
# from torch.nn.utils.rnn import pad_packed_sequence

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = "wiki10", choices = DATASET)
parser.add_argument('--K', type = int, default = 20) # deli:11   wiki: 5 deli:10
parser.add_argument('--L',type = int, default = 10)   # wiki: 5 deli 10
parser.add_argument('--rebuild_freq',type = int, default = 20)  #wiki:30
parser.add_argument('--epoch_num', type=int, default=30)
parser.add_argument('--lr_task', type=float, default=0.0001 )
parser.add_argument('--lr', type=float, default=0.001 )
parser.add_argument('--batch_size', type=int, default=128)    # 128
parser.add_argument('--layer_dim',type = int, default = 128)    
parser.add_argument('--margin', type = float, default = 1)
parser.add_argument('--seed',type = int, default = 17)
parser.add_argument('--print_every', type = int, default = 100)   
parser.add_argument('--test_every', type = int, default = 100)
parser.add_argument('--triplet_flag', type = bool, default = True)
parser.add_argument('--reset', type = bool, default = False)
#parser.add_argument('--num_processes', type=int, default=20
#parser.add_argument('--cuda_device',type = str, default = "1")

args = parser.parse_args()
now = datetime.now()
log_time = date_time = now.strftime("%m%d%H%M%S")
logfile = "./inference_slide_log/{}/K{}_L{}_r{}_b{}_at{}.txt".format( args.dataset,args.K, args.L, args.rebuild_freq, args.batch_size,log_time)
print("args",args,file = open(logfile, "a"))
print("args",args)


def get_networkDataLoader(args):
    # set up dataset object
    train_ds = MultiLabelDataset( DATAPATH_TRAIN[args.dataset])
    test_ds = MultiLabelDataset( DATAPATH_TEST[args.dataset])
    # feed dataset to create dataloader
    # num_works != 0 -> semaphore_Tracker, segmentation fault
    train_ld = DataLoader( train_ds, pin_memory = True,num_workers = 8, shuffle = True, batch_size = args.batch_size)
    test_ld = DataLoader( test_ds, pin_memory = True,num_workers = 8, shuffle = False, batch_size =256)
    #test_ld = DataLoader( test_ds, pin_memory = True,num_workers = 0, shuffle = True, batch_size = args.batch_size)

    return train_ld, test_ld, train_ds.D, train_ds.L, train_ds.N, test_ds.N

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)        

def train_network(args, model, device, train_loader, test_loader, optimizer, epoch, freeze_flag):
    print("freeze_flag:",freeze_flag )
    model.train()

    epoch_start_time = time.time()

    for idx, (y, x) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        logits, new_targets, nsamples = model.forward(x, y, freeze_flag)

        output_dist = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(output_dist, new_targets, reduction='sum') / args.batch_size 

        loss.backward()
        optimizer.step()
        #print train loss
        if idx % args.print_every == args.print_every-1:  # print every 100 mini-batches
            print('===[%d, %5d] Train loss: %.3f, table load: %.3f' % (epoch , idx + 1, loss.item(),model.lshLayer.lsh.stats()))
            
            print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
            # print("inner product: sampe - random: ", s_r_ip)
        
        #print evaluate accuracy
        if idx % args.test_every  == args.test_every -1 :  
            evaluate(args,model,device,test_loader,freeze_flag,k = 5, training = True)

        if(freeze_flag == False):
        #rebuild hash table
            if(idx % args.rebuild_freq == 0 and idx!= 0 ):
                print(r"rebuild hash table")
                model.lshLayer.buildLSH()

        # torch.cuda.empty_cache()

def evaluate(args, model, device, loader, freeze_flag = False ,k=1 ,training = False):
    model.eval()
    N = 0.
    N_1 = 0.
    correct = 0.
    top1 = 0.
    with torch.no_grad():
        for batch_idx, (labels, data) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)
            output,_,_ = model.forward(data, labels,freeze_flag)
            output = output.cpu()

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if indices[bdx, idx].item() in label_set:
                        correct+=1.

            values, indices = torch.topk(output, 1, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(1):
                    N_1 += 1
                    if indices[bdx, idx].item() in label_set:
                        top1+=1.
            if(batch_idx == 20 and training):
                # print("predicte",indices)
                break
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(1,top1/N_1, top1))
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(k,correct/N, correct))
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(1,top1/N_1, top1),file = open(logfile, "a"))
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(k, correct/N, correct),file = open(logfile, "a"))

    model.train()

            

if __name__ == "__main__":

    #set up seed
    np.random.seed(1234)
    torch.manual_seed(1234)

    print("device",device)
    #read in train and test data
    print("\n===========Read in data: " + args.dataset + "===================")
    train_loader, test_loader, feature_dim, num_class, num_train, num_test = get_networkDataLoader(args)
    print("Dataset Statistics: feature dimension: %d, label dimension: %d,  number of train data: %d, number of test data: %d"
        %(feature_dim, num_class, num_train, num_test))

    print("\n============Set Up Network==================")
    model = Net(feature_dim, num_class, args.layer_dim, args.K, args.L).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr_task)
    # optimizer = optim.SparseAdam(model.parameters(), lr = args.lr)


    freeze_flag = False
    print("\n============Training start=================")
    for epoch in range(args.epoch_num):

        if(epoch > 5):
            freeze_flag = True

        print("Epoch: ", epoch)
        epoch_start_time = time.time()
        train_network(args, model, device, train_loader, test_loader,optimizer, epoch, freeze_flag )
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
        evaluate(args,model,device,test_loader,freeze_flag,5)


    torch.save(model.state_dict(), "./same_model/deli/model")
        







        




    






