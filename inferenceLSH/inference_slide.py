import os
import sys
import numpy as np
import argparse
from datetime import datetime

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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

DATASET = ['wiki10', 'amz13k','amz630k','deli']
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = "deli", choices = DATASET)
parser.add_argument('--K', type = int, default = 11) #13   wiki: 5 deli:10
parser.add_argument('--L',type = int, default = 10)   # wiki: 5
parser.add_argument('--rebuild_freq',type = int, default = 30)
parser.add_argument('--epoch_num', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001 )
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layer_dim',type = int, default = 128)
parser.add_argument('--margin', type = float, default = 1)
parser.add_argument('--seed',type = int, default = 17)
parser.add_argument('--print_every', type = int, default = 109)
parser.add_argument('--test_every', type = int, default = 1000)
#parser.add_argument('--num_processes', type=int, default=20)
#parser.add_argument('--cuda_device',type = str, default = "1")

args = parser.parse_args()
now = datetime.now()
time = date_time = now.strftime("%m%d%H%M%S")
logfile = "./inference_slide_log/{}/K{}L{}r{}b{}at{}.txt".format( args.dataset,args.K, args.L, args.rebuild_freq, args.batch_size,time)
print("args",args,file = open(logfile, "a"))

'''
Helpter Function
'''
def get_networkDataLoader(args):
    # set up dataset object
    train_ds = MultiLabelDataset( DATAPATH_TRAIN[args.dataset])
    test_ds = MultiLabelDataset( DATAPATH_TEST[args.dataset])
    # feed dataset to create dataloader
    # num_works != 0 -> semaphore_Tracker, segmentation fault
    train_ld = DataLoader( train_ds, pin_memory = True,num_workers = 0, shuffle = False, batch_size = args.batch_size)
    test_ld = DataLoader( test_ds, pin_memory = True,num_workers = 0, shuffle = False, batch_size =args.batch_size)
    return train_ld, test_ld, train_ds.D, train_ds.L, train_ds.N, test_ds.N

def get_tripletDataLoader(train_data, num_iter, batch_size = 16, shuffle = True, pin_memory=False):
    train_Dataset = Dataset(train_data)
    train_dataloader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)
    return train_dataloader

def getTripletWeight(triplet_dict):
    hash_weight = torch.empty(1)
    for l,triplet in triplet_dict.items():
        if(l == 0):
            hash_weight = triplet.classifier.dense1.weight
        else:
            hash_weight = torch.cat( (hash_weight,triplet.classifier.dense1.weight), 0)
    hash_weight = torch.t(hash_weight)
    return hash_weight

def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def unpackbits(x, num_bits):
    x = x.reshape(-1, 1).int()
    to_and = 2 ** torch.arange(num_bits).reshape(1,num_bits).int()
    return (x & to_and).bool().int()


'''
Training Related
'''
def train_cl(id, triplet_data, net_loss, opt_cl1, triplet,  s_hashcode, device, num_iter):
    opt_cl1.zero_grad()
    triplet.to(device)
    x = triplet_data['arc']
    p = triplet_data['pos']
    n = triplet_data['neg']
    x, p, n = x.to(device).float(), p.to(device).float(), n.to(device).float()

    running_loss = 0.0

    s_hashcode = torch.tensor(s_hashcode)
    #convert hash code to binary representation
    binary = unpackbits( s_hashcode, args.K ).to(device).float()
    load_weight = (torch.abs(torch.sum(binary,dim =0) - args.batch_size/2) + 0.01 ) / args.batch_size
   

    triplet_loss, emb = triplet.forward(x, p, n)

    # task loss = (1-y)*log(prob) + y*log(prob)
    newx = x.repeat(1, args.K).view(-1, x.size()[1])
    newweight = triplet.classifier.dense1.weight.data.repeat(x.size()[0], 1)
    prob = (1 - torch.acos(F.cosine_similarity(newx, newweight).view(-1, args.K))/3.141592653 ).to(device)

    #print( (1 - binary) * torch.log(1 - prob + 1e-8 ) + (binary) * torch.log(prob + 1e-8) )
    #print( -net_loss  * ( (1 - binary) * torch.log(1 - prob + 1e-8 ) + (binary) * torch.log(prob + 1e-8) ))
    taskloss =  torch.mean( -net_loss * ( (1 - binary) * torch.log(1 - prob + 1e-8 ) + (binary) * torch.log(prob + 1e-8) ) )
    
    loadloss = torch.mean(load_weight * ( (1 - binary) * torch.log(1 - prob + 1e-8 ) + (binary) * torch.log(prob + 1e-8) ) )

    #combine loss
    # print((1 - binary) * torch.log(1 - prob + 1e-8 ) + (binary) * torch.log(prob + 1e-8))
    # print(net_loss)
    # print("\ntaskloss",taskloss )
    # print("triplet_loss",triplet_loss)
    # print("loadloss",loadloss)
   
    tripletloss_weight = 0.1    #use to enable/disbale two loss      
    taskloss_weight = 1      #use to enable/disbale two loss
    load_weight = 0
    loss = taskloss *  taskloss_weight + triplet_loss * tripletloss_weight + loadloss * load_weight
    # print("train_cl loss:", loss.item())

    loss.backward()
    opt_cl1.step()

    # print statistics
    running_loss += loss.item()
    return running_loss, taskloss
        

def train_network(args, model, device, train_loader, test_loader,optimizer, epoch, triplet_dict, triplet_opt1_dict,triplet_opt2_dict):
    #test before train start
    #evaluate(args,model,device,test_loader,1,True)

    model.train()

    #use triplet network weight as hash table
    triplet_flag = True

    triplet_baseline = []
    avg_triplet_baseline = 0.0
    std_triplet_baseline = 1

    for idx, (y, x,x_v) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        logits, new_targets, nsamples, weight_pair,sid, s_hashcode, s_ip, s_r_ip= model.forward(x,x_v,y)
        output_dist = F.log_softmax(logits, dim=-1)

        loss = F.kl_div(output_dist, new_targets, reduction='sum') / args.batch_size 
        #loss = F.binary_cross_entropy_with_logits(logits, new_targets, reduction='sum') / args.batch_size 
        #loss = F.binary_cross_entropy_with_logits(logits.view(-1, nsamples), new_target, reduction='sum') / batch_size
        loss.backward()
        optimizer.step()

        #reset model last extra weight, used for padding
        model.weights.data[-1,:].zero_()
        model.lshLayer.params.weight.data[-1,:].zero_()

        #print train loss
        if idx % args.print_every == args.print_every-1:  # print every 100 mini-batches
            print('===[%d, %5d] Train loss: %.3f, table load: %.3f' % (epoch , idx + 1, loss.item(),model.lshLayer.lsh.stats()))
            print("inner product: sampe - random: ", s_r_ip)
            
        
        #print evaluate accuracy
        if idx % args.test_every  == args.test_every -1 :  
            evaluate(args,model,device,test_loader,k = 1,training = True)
            evaluate(args,model,device,test_loader,k = 5,training = True)
        
        #collect data for triplet network
        weight_pair['arc'] = weight_pair['arc'].detach()
        weight_pair['pos'] =weight_pair['pos'].detach()
        weight_pair['neg'] =weight_pair['neg'].detach()

        to_triplet_loss = F.kl_div(output_dist, new_targets, reduction = 'none')
        to_triplet_loss = torch.sum(to_triplet_loss, dim = 1).view(-1,1)
        to_triplet_loss = ((to_triplet_loss- torch.mean(to_triplet_loss))/torch.std(to_triplet_loss)).detach()
        
        #processes = []
        print_loss = 0
        for l in range(args.L):
            triple_loss, baseline = train_cl(l, weight_pair, to_triplet_loss, triplet_opt1_dict[l],triplet_dict[l], s_hashcode[:,l], device, idx)
            print_loss += triple_loss

            # in case parellep on cpu to train triplet
            # p = mp.Process(target=train_cl, args=(l, weight_pair, to_triplet_loss, triplet_opt1_dict[l],triplet_dict[l], s_hashcode[:,l], device, idx))
            # p.start()
            # processes.append(p)

        # for p in processes:
        #     p.join()

        #whitening loss
        # triplet_baseline += [loss.item()]
        # avg_triplet_baseline = np.mean(np.array(triplet_baseline))
        # std_triplet_baseline = np.std(np.array(triplet_baseline))

        if idx % args.print_every == args.print_every-1:  # print every 100 mini-batches
            print('===[%d, %5d] Triplet Train loss: %.3f' % (epoch , idx + 1, print_loss / args.L))


        #rebuild hash table
        if(idx % args.rebuild_freq == 0 and idx!= 0 ):
            #print("====[%d, %5d] Rebuild Hash table"% (epoch , idx + 1))
            #if flag is true, using triplet weight to set simhash
            if(triplet_flag):
                model.lshLayer.hash_weight = getTripletWeight(triplet_dict)
                model.lshLayer.buildLSH(True)
            else:
                model.lshLayer.buildLSH(False)

        torch.cuda.empty_cache()

    evaluate(args,model,device,test_loader,1,False)
    evaluate(args,model,device,test_loader,5,False)

            

def evaluate(args, model, device, loader, k=1,training=False):
    model.eval()
    N = 0.
    correct = 0.
    with torch.no_grad():
        for batch_idx, (labels, data, value) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)
            output = model(data, value, labels).cpu()
            # values, indices = torch.max(output, dim=1)
            #print("predicte",indices)

            # for bdx in range(batch_size):
            #     N += 1
            #     label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
            #     if indices[bdx].item() in label_set:
            #         correct+=1
            # #print("num of correct",correct)
            # #print(N)

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if indices[bdx, idx].item() in label_set:
                        correct+=1.
                        # if idx == 0:
                        #     top1+=1.
            
            if(batch_idx == 20 and training):
                # print("predicte",indices)
                break
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(k,correct/N, correct))
    print("{}===Test Accuracy {:.4f}, total_correct {}".format(k, correct/N, correct),file = open(args.logfile, "a"))
    model.train()

            

if __name__ == "__main__":
    # args = parser.parse_args()

    #set up seed
    np.random.seed(args.seed)
    torch.manual_seed(0)

    #set up cuda
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,5,6"
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:1" if use_cuda else "cpu")

    mp.set_start_method('spawn')

    print("device",device)

    #read in train and test data
    print("\n===========Read in data: " + args.dataset + "===================")
    train_loader, test_loader, feature_dim, num_class, num_train, num_test = get_networkDataLoader(args)
    print("Dataset Statistics: feature dimension: %d, label dimension: %d,  number of train data: %d, number of test data: %d"
        %(feature_dim, num_class, num_train, num_test))

    #y,x = next(iter(train_loader))
    #print(x)

    # set up triplet network 
    print("\n===========Set Up Triplet Network================")
    triplet_dict = {}
    triplet_opt1_dict = {}
    triplet_opt2_dict = {}
    for l in range(args.L):
        classifier = Network(args.layer_dim, args.K)
        triplet_dict[l] = TripletNet(classifier, args.margin)
        triplet_opt1_dict[l] = optim.SGD(triplet_dict[l].parameters(), lr = args.lr, momentum = 0.9)
        triplet_opt2_dict[l] = optim.SGD(triplet_dict[l].parameters(), lr = args.lr, momentum = 0.9)
        #triplet_dict[l].to(device)
        # print("classifier %d"%(l))
        # print("network weight0 shape:", triplet_dict[l].classifier.dense1.weight.shape)
    # collect weight for hash table
    hash_weight = getTripletWeight(triplet_dict).cpu()
    print("hash weight shape:", hash_weight.shape)
    print("\n============Set Up Network==================")
    model = Net(feature_dim, num_class, args.layer_dim, hash_weight, args.K, args.L).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)


    print("\n============Training start=================")
    with open(args.logfile,'w') as out:
        for epoch in range(args.epoch_num):
            print("Epoch: ", epoch)
            train_network(args, model, device, train_loader, test_loader,optimizer, epoch,triplet_dict,triplet_opt1_dict,triplet_opt2_dict)

            
        







        




    






