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

import numpy as np
from lazy_parser import MultiLabelDataset
from torch.utils.tensorboard import SummaryWriter
from adam_base import Adam
from lsh_softmax import LSHSoftmax
from config import config
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

# DATASET = ['wiki10', 'amz13k','amz630k','deli','wiki300']
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
parser.add_argument('--dataset', type = str, default = "wiki10", choices = DATASET)
parser.add_argument('--K', type = int, default = 20) 
parser.add_argument('--L',type = int, default = 20)   
parser.add_argument('--rebuild_freq',type = int, default = 30)  #wiki:30
parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='input batch size for training (default: 1)')
parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N',help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default= '0.005', metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--seed', type=int, default=4321, metavar='S', help='random seed (default: 1)')
parser.add_argument('--print_every', type = int, default = 100)
parser.add_argument('--test_every', type = int, default = 100)

parser.add_argument('--resume_full',type = bool, default = False)
parser.add_argument('--resume_model_path', type = str, default = "")
args = parser.parse_args()
print('args:',parser.parse_args())

best_acc1 = 0.0
best_acc5 = 0.0
model_checkfile = "./checkpoint/{}/model_K{}_L{}_batch{}_testbatch{}_lr{}.pth.tar".format( args.dataset,args.K, args.L, args.batch_size,args.test_batch_size, args.lr)
logfile = "./inference_slide_log/{}/K{}_L{}_b{}_testbatch{}.txt".format( args.dataset,args.K, args.L, args.batch_size,args.test_batch_size)
print("args",args,file = open(logfile, "a"))

runfile = "./runs/{}_K{}_L{}_r{}_b{}_lr{}_ryan".format( args.dataset,args.K, args.L, args.rebuild_freq, args.batch_size, args.lr)
print("\nTensorboard file: ", runfile)
writer = SummaryWriter(runfile)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    print("saved better model")

class Net(nn.Module):
    def __init__(self, IN, OUT):
        super(Net, self).__init__()
        self.IN = IN
        self.OUT = OUT
        self.fc = nn.Embedding(self.IN+1, 128, padding_idx=IN, sparse=True)

        self.bias = nn.Parameter(torch.Tensor(128))
        stdv = 1. / math.sqrt(self.IN)
        self.bias.data.uniform_(-stdv, stdv)

        self.smax = LSHSoftmax(N=OUT, D=128, K=args.K, L=args.L, freq=30)
        # self.lshLayer = LSHSoftmax(N=OUT, D=128, K=args.K, L=args.L, freq=30)
        #self.smax = SampledSoftmax(ntokens=OUT, nsampled=OUT//6, nhid=128)

    def forward(self, x, y,freeze,slide):
        raw_emb = torch.sum(self.fc(x), dim=1)
        emb = raw_emb / torch.norm(raw_emb, dim=1, keepdim=True)
        query = F.relu(emb + self.bias)
        return self.smax.forward(query, y,freeze,slide)

def train(args, model, device, loader, test_loader, optimizer, epoch, freeze):
    slide = False
    model.train()
    start_time = time.time()
    for batch_idx, (labels, data) in enumerate(loader):
        step = epoch * len(loader)+ batch_idx

        optimizer.zero_grad()
        data = data.to(device)

        logits, new_targets, nsamples = model.forward(data, labels, freeze,slide)
        output_dist = F.log_softmax(logits.view(-1, nsamples), dim=-1)
        batch_size = labels.size(0)
        loss = F.kl_div(output_dist, new_targets, reduction='sum') / batch_size
        
        t1 = time.time()
        loss.backward()
        optimizer.step()

        if(freeze==False):
            if(batch_idx % args.rebuild_freq == 0 and batch_idx!= 0 ):
                print("rebuild hash table")
                model.smax.build()

        if batch_idx % args.print_every == args.print_every-1: 
            time_passed = time.time() - start_time
            print('===[%d, %5d] Network Train -> loss: %.3f' % (epoch , batch_idx + 1, loss.item()))
            print('===[%d, %5d] Time: %.5fs |'% (epoch, batch_idx + 1, time_passed))
            writer.add_scalar('task_loss/train', loss.item(),step)        

        # if batch_idx % args.test_every == args.test_every-1: 
        #     s_top1, s_top5 = evaluate_slide(args, model, device, test_loader, training = True, k=5, slide = True)
        #     top1, top5 = evaluate(args, model, device, test_loader, training = True, k=5,slide = False)
        #     slide_better, full_better = analysis(args, model, device, test_loader)
        #     writer.add_scalar('inference/slide_top1', s_top1,step)
        #     writer.add_scalar('inference/slide_top5', s_top5,step)
        #     writer.add_scalar('inference/full_top1', top1,step)
        #     writer.add_scalar('inference/full_top5', top5,step)
        #     writer.add_scalar('better/slide_better', slide_better,step)
        #     writer.add_scalar('better/full_better', full_better,step)

    # end of epoch testing, analysis
    slide_better, full_better = analysis(args, model, device, test_loader)
    s_top1, s_top5 = evaluate_slide(args, model, device, test_loader, training = False, k=5, slide = True)
    # top1, top5 = evaluate(args, model, device, test_loader, training = False, k=5, slide = False)
    writer.add_scalar('inference/slide_top1', s_top1,step)
    writer.add_scalar('inference/slide_top5', s_top5,step)
    # writer.add_scalar('inference/full_top1', top1,step)
    # writer.add_scalar('inference/full_top5', top5,step)
    writer.add_scalar('better/slide_better', slide_better,step)
    writer.add_scalar('better/full_better', full_better,step)


def evaluate_slide(args, model, device, loader, training, k=1,slide=False):
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
            output, sample_size,id_list = model(data,labels,freeze,slide)
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
    print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
    print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
    print("[Slide: {}] Sample size: {}".format(slide, sample_size))
    print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
    print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))
    print("[Slide: {}] Sample size: {}".format(slide, sample_size),file = open(logfile, "a"))
    

    model.train()
    return top1_acc, topk_acc

def evaluate(args, model, device, loader, training, k=1,slide=False):
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
            output = model(data,labels,freeze,slide).cpu()

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if indices[bdx, idx].item() in label_set:
                        correct+=1.
                        if idx == 0:
                            top1 += 1

            if( batch_idx == stop and training ):
                break

    top1_acc = top1/N * k
    topk_acc = correct/N
    print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1))
    print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
    print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
    print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))

    model.train()
    return top1_acc, topk_acc

def analysis(args, model, device, loader, k=1):
    freeze = False
    model.eval()
    slide_do_better=0
    full_do_better=0
    N = 0.
    correct = 0.
    top1 = 0.
    stop=int(10000/args.test_batch_size)
    with torch.no_grad():
        for batch_idx, (labels, data) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.to(device)
            output_full = model.forward(data,labels,freeze,False).cpu()
            values_full, indices_full = torch.topk(output_full, k=k, dim=1)

            output, sample_size,id_list = model.forward(data,labels,freeze,True)
            output=output.cpu()
            values, indices = torch.topk(output, k=k, dim=1)

            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if id_list[indices[bdx, idx].item()] in label_set:
                        if indices_full[bdx, idx].item() not in label_set:
                            slide_do_better+=1
                    if indices_full[bdx, idx].item() in label_set: 
                        if id_list[indices[bdx, idx].item()] not in label_set:
                            full_do_better+=1

            if batch_idx == stop:
                break

    print("SLIDE do better: {}, Full do better {}".format(slide_do_better, full_do_better))
    print("SLIDE do better: {}, Full do better {}".format(slide_do_better, full_do_better),file = open(logfile, "a"))
    # print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct))
    # print("[Slide: {}] Top{}:===Test Accuracy {:.4f}, total_correct {}".format(slide, 1,top1_acc, top1),file = open(logfile, "a"))
    # print("[Slide: {}] Top{}: ===Test Accuracy {:.4f}, total_correct {}".format(slide, k,topk_acc, correct),file = open(logfile, "a"))

    model.train()
    return slide_do_better, full_do_better
    # return top1_acc, topk_acc

def main():
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = MultiLabelDataset(DATAPATH_TRAIN[args.dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=True)

    test_dataset = MultiLabelDataset(DATAPATH_TEST[args.dataset])
    #test_dataset = MultiLabelDataset(DATAPATH_TRAIN[args.dataset])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=6, shuffle=False)
    
    print("Statistics:", train_dataset.N, train_dataset.D, train_dataset.L, train_dataset.max_D, train_dataset.max_L)
    model = Net(train_dataset.D, train_dataset.L).to(device)
    optimizer = Adam(model.parameters(args.lr))
    
    freeze = False
    best_acc1 = 0.0
    best_acc5 = 0.0
    start_epoch = 0
    resume_path=args.resume_model_path
    if(args.resume_full):
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path)) #if resume from saved model


    for epoch in range(start_epoch, args.epochs, 1):
        print("epoch {}, freeze {}:".format(epoch, freeze))

        epoch_start_time = time.time()
        train(args, model, device, train_loader,test_loader, optimizer, epoch, freeze)

        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
    
        top1_acc,top5_acc = evaluate(args, model, device, test_loader, training = False, k=5,slide = False)

        is_best = (top1_acc > best_acc1) or (top5_acc > best_acc5)
        best_acc1 = max(  top1_acc,best_acc1)
        best_acc5 = max(  top5_acc,best_acc5)

        for p in model.parameters():
            if p.requires_grad:
                print(p.name,p.size())

        if(best_acc1> FREEZE_ACC[args.dataset] and freeze == False):
            freeze=True
            #freeze embedding
            print("Set freeze weight")
            model.fc.weight.requires_grad = False
            model.bias.requires_grad = False
            print("Current Parameters")
            for p in model.parameters():
                if p.requires_grad:
                    print(p.name,p.size())
        
        


        print('-' * 89)
        print()

if __name__ == '__main__':
    main()
