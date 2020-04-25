import torch
torch.manual_seed(0)
from torch import nn
import torch.multiprocessing as mp
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from os import path
import os

from model import FCNSG,LRSG
from dataset import get_dataset
import pdb
import argparse
import time
from os.path import dirname, abspath, join
import glob
cur_dir = dirname(abspath(__file__))



parser = argparse.ArgumentParser()
parser.add_argument('--async', action="store", dest="asyn", type=bool, default=False,
                    help="Type True/False to turn on/off the async SGD. Default False")
parser.add_argument("--process", action="store", dest="process", type=int, default=4,
                    help="Number of processes to use if asynchronous SGD is turned on. Default 4")
parser.add_argument('--MH', action="store_true", default=False,
                    help="to use MinHash/feature hashing files as input. Default False")
parser.add_argument('--K', action="store", dest="K", type=int, default=1000,
                    help="K minhashes to use. The corresponding minhash file should be generated already. Default 1000")
parser.add_argument('--L', action="store", dest="L", type=int, default=3,
                    help="L layers of fully connected neural network to use. Default 3")
parser.add_argument('--dataset', action="store", dest="dataset", default="url",
                    help="Dataset folder to use. Default url")
parser.add_argument('--epoch', action="store", dest="epoch", type=int, default=10,
                    help="Number of epochs for training. Default 10")
parser.add_argument('--batch', action="store", dest="batch_size", type=int, default=1024,
                    help="Batch size to use. Default 1024")
parser.add_argument('--reduced_dimension', action="store", dest="reduced_dimension", type=int, default=3231961,
                    help="Dimension reduction by FH or MH")
parser.add_argument('--bbits', action="store", dest="bbits", type=int, default=8, 
                    help="number of bits to store for MH")
parser.add_argument('--pairwise', action="store_true", default=False,
                    help="to use pairwise data / simple data. Default False")
parser.add_argument('--hashfull', action="store_true", default=False,
                    help="hashfull empty bins in DMH. Default False")
parser.add_argument('--use_mh_computation', action="store", default="univ",
                    help="univ: vectorised 4 universal, rotdense : rotation densified, orig: original")
parser.add_argument('--device', action="store", dest="device", type=int, default=0, 
                    help="ID of GPU")
parser.add_argument('--lr', action="store", dest="lr", type=float, default=0.0001, 
                    help="Learning rate")
parser.add_argument('--use_classwts', action="store_true", default=False,
                    help="Use class wts if avaiable")
parser.add_argument('--weight_decay', action="store", dest="weight_decay", type=float, default=0, 
                    help="l2 penatly default 0")
parser.add_argument('--save_model_itr', action="store", dest="save_model_iteration", type=int, default=1000000, 
                    help="% Iterations at which we should store the model")
parser.add_argument('--eval_model_itr', action="store", dest="eval_model_iteration", type=int, default=1000000, 
                    help="% Iterations at which we should store the model")
parser.add_argument('--load_latest_ckpt', action="store_true", default=False,
                    help="load latest ckpt")

results = parser.parse_args()


# ===========================================================
# Global variables & Hyper-parameters
# ===========================================================
DATASET = results.dataset
ASYNC = results.asyn
PROCESS = results.process
MH = results.MH
D = results.reduced_dimension
K = results.K
bbits = results.bbits
L = results.L
EPOCH = results.epoch
BATCH_SIZE = results.batch_size
GPU_IN_USE = True  # whether using GPU
PAIRWISE = results.pairwise
device_id = results.device
HASHFULL = results.hashfull
MHCOMPUTATION = results.use_mh_computation
LRATE = results.lr
USECLASSWT = results.use_classwts
WEIGHT_DECAY = results.weight_decay
SAVE_MODEL_ITERATION = results.save_model_iteration
LOAD_LATEST_CKPT = results.load_latest_ckpt
EVALUATE_ITR = results.eval_model_iteration
class_weights = None
if USECLASSWT:
  if DATASET == "avazu":
    class_weights = torch.tensor([0.566, 4.266], dtype=torch.double)
    print(DATASET, ": Using class weights", class_weights)



def train(data_files, dim, model, MHTrain, time_file=None, record_files=None, p_id=None):
    # ===========================================================
    # Prepare train dataset & test dataset
    # ===========================================================
    print("***** prepare data ******")
    train, train_small, test, test_small = data_files
    if record_files is not None:
        acc_name, valacc_name, loss_name, valloss_name,final_prediction_name,checkpoint_name,index_name = record_files

    training_set = get_dataset(train, dim, MHTrain, K, PAIRWISE, HASHFULL, MHCOMPUTATION)
    train_dataloader = torch.utils.data.DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_set = get_dataset(test, dim, MHTrain, K, PAIRWISE, HASHFULL, MHCOMPUTATION)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=False)

    training_set_small = get_dataset(train_small, dim, MHTrain, K, PAIRWISE, HASHFULL, MHCOMPUTATION)
    train_dataloader_small = torch.utils.data.DataLoader(dataset=training_set_small, batch_size=BATCH_SIZE, shuffle=True)
    validation_set_small = get_dataset(test_small, dim, MHTrain, K, PAIRWISE, HASHFULL, MHCOMPUTATION)
    validation_dataloader_small = torch.utils.data.DataLoader(dataset=validation_set_small, batch_size=BATCH_SIZE, shuffle=False)

    print("***** prepare optimizer ******")
    optimizer = torch.optim.Adam(model.parameters(), lr=LRATE, weight_decay=WEIGHT_DECAY)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_func = nn.BCELoss(weight=class_weights).cuda(device_id) if GPU_IN_USE else nn.BCELoss(weight=class_weights)
    plain_loss_func = nn.BCELoss().cuda(device_id) if GPU_IN_USE else nn.BCELoss()
    epoch = 0
    
    if LOAD_LATEST_CKPT :
      files=glob.glob(checkpoint_name + "*")
      if len(files) > 0:
          files.sort(key=path.getmtime)
          lcheckpoint_name = files[-1]
          print("Loading from checkpoint", lcheckpoint_name)
          checkpoint = torch.load(lcheckpoint_name)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          epoch = checkpoint['epoch']
          print("Epoch:",epoch)
          print("Iteration:", checkpoint["iteration"])
      else:
          print("CHECKPOINT NOT FOUND")
          return
    else:
      # clear record files
      for f in record_files:
        if path.isfile(f):
          print("removing",f)
          os.remove(f)

    print("***** Train ******")
    acc_list, valacc_list = [], []
    loss_list, valloss_list = [], []
    index_list = []
    training_time = 0.

    while epoch < EPOCH :
        # Training
        for iteration, (x, y) in enumerate(train_dataloader):

            if iteration % SAVE_MODEL_ITERATION == 0 and iteration > 0:
                torch.save({
                    'epoch': epoch,
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_name + ".latest")
            if (epoch == 0 and iteration in [0] + [40*2**i for i in range(0, 25)]) or (epoch > 0 and iteration % EVALUATE_ITR == 0): # in epoch 0: run at log iterations and after that , run at every %20K [0, 20,K,40K etc]
                index_list.append(np.array([epoch, iteration], dtype=int))
                train_acc, train_loss = validation(model, train_dataloader_small, plain_loss_func, full=True)
                acc_list.append(train_acc)
                loss_list.append(train_loss.data)
                print("*" * 50)
                print('## Epoch: ', epoch, '| Iteration: ', iteration, '| total train loss: %.4f' % train_loss.data,
                          '| total train accuracy: %.2f' % train_acc)

                valid_acc, valid_loss = validation(model, validation_dataloader, plain_loss_func, full=True, prediction_file=final_prediction_name+"_E"+str(epoch)+"_IT"+str(iteration), write=True)
                valacc_list.append(valid_acc)
                valloss_list.append(valid_loss)
                print('## Epoch: ', epoch, '| Iteration: ', iteration, '| total validation loss: %.4f' % valid_loss,
                          '| total validation accuracy: %.2f' % valid_acc)
                print("*" * 50)
                if record_files is not None:
                    with open(acc_name, "ab") as f_acc_name, open(valacc_name, "ab") as f_valacc_name, open(loss_name, "ab") \
                          as f_loss_name, open(valloss_name, "ab") as f_valloss_name, open(index_name, "ab") as f_index_name:
                        np.savetxt(f_acc_name, acc_list)
                        np.savetxt(f_valacc_name, valacc_list)
                        np.savetxt(f_loss_name, loss_list)
                        np.savetxt(f_valloss_name, valloss_list)
                        np.savetxt(f_index_name, index_list, fmt="%d")
                        f_acc_name.close()
                        f_valacc_name.close()
                        f_loss_name.close()
                        f_valloss_name.close()
                        f_index_name.close()
                        acc_list = []
                        valacc_list = []
                        loss_list = []
                        valloss_list = []
                        index_list = []

            start = time.clock()
            model.train()
            x = Variable(x).cuda(device_id) if GPU_IN_USE else Variable(x)
            y = Variable(y).cuda(device_id) if GPU_IN_USE else Variable(y)
            y = y.double()
            output = model(x)
            y = y.reshape(output.shape[0], 1)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_time += time.clock() - start
            predicted = output.data > 0.5
            train_accuracy = (predicted == y.data.bool()).sum().item() / y.data.shape[0]

            if iteration % 100 == 0:
                print('Epoch: ', epoch, '| Iteration: ', iteration, '| batch train loss: %.4f' % loss.data,
                          '| batch train accuracy: %.2f' % train_accuracy)
            


        #scheduler.step()
        # Saving records
        epoch = epoch + 1
        torch.save({
          'epoch': epoch,
          'iteration': 0,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_name + ".latest")


    valid_acc, valid_loss = validation(model, validation_dataloader, plain_loss_func, full=True, prediction_file=final_prediction_name+"_E"+str(epoch)+"_IT0", write=True)
    print('Final validation loss: %.4f' % valid_loss,
                          '| final validation accuracy: %.2f' % valid_acc)

    valacc_list.append(valid_acc)
    valloss_list.append(valid_loss)
    index_list.append(np.array([epoch, 0], dtype=int))
    with open(acc_name, "ab") as f_acc_name, open(valacc_name, "ab") as f_valacc_name, open(loss_name, "ab") \
          as f_loss_name, open(valloss_name, "ab") as f_valloss_name, open(index_name, "ab") as f_index_name:
        np.savetxt(f_acc_name, acc_list)
        np.savetxt(f_valacc_name, valacc_list)
        np.savetxt(f_loss_name, loss_list)
        np.savetxt(f_valloss_name, valloss_list)
        np.savetxt(f_index_name, index_list, fmt="%d")
        f_acc_name.close()
        f_valacc_name.close()
        f_loss_name.close()
        f_valloss_name.close()
        f_index_name.close()
        acc_list = []
        valacc_list = []
        loss_list = []
        valloss_list = []
        index_list = []


    if time_file is not None:
        with open(time_file, 'a+') as outfile:
            prefix = "(ASYNC, id={}) ".format(p_id) if ASYNC else ""
            if MH:
                outfile.write("{}K={},   L={}, epoch={} | time={}\n".format(prefix, K, L, EPOCH, training_time))
            else:
                outfile.write("{}Baseline, L={}, epoch={} | time={}\n".format(prefix, L, EPOCH, training_time))


def validation(model, validation_dataloader, plain_loss_func, full=False, prediction_file=None, write=False):
    count = 0
    total = 0.
    valid_correct = 0.
    total_loss = 0.
    model.eval()
    random_mod=np.random.randint(10)+1
    if full and write:
      if path.isfile(prediction_file):
            print("Found prediction file present. Removing :",prediction_file)
            os.remove(prediction_file)
      f = open(prediction_file, "ab")
      print("full operation: writing predictions to ", prediction_file)
      
    with torch.no_grad():
        for batch_id, (x_t, y_t) in enumerate(validation_dataloader):
            if batch_id % random_mod==0 or full:
                x_t = Variable(x_t).cuda(device_id) if GPU_IN_USE else Variable(x_t)
                y_t = Variable(y_t).cuda(device_id) if GPU_IN_USE else Variable(y_t)
                y_t = y_t.double()
                output = model(x_t)
                predicted = output.data > 0.5

                y_t = y_t.reshape(output.shape[0], 1)
                loss = plain_loss_func(output, y_t)
                total_loss += loss
                valid_correct += (predicted == y_t.data.bool()).sum().item()
                total += y_t.data.shape[0]
                count += 1
                if full and write:
                    np.savetxt(f, X=output.cpu().data.numpy()[:,0], fmt="%1.6f")
                    f.flush()
                #print("Loss", total_loss / count)
            if count > 50 and (not full):
              break;
        if full and write:
          f.close()
        valid_accuracy = valid_correct / total
        valid_loss = total_loss / count

    return valid_accuracy, valid_loss


if __name__ == '__main__':
    print("dataset={}; async={}; num_process={} ; MH={}; K={}; L={}; epoch={}; batch_size={}".format(DATASET, ASYNC, PROCESS, MH, K, L, EPOCH, BATCH_SIZE))

    #########################################
    print("***** prepare model ******")
    if L == 0:
        model = LRSG(dimension=D).double()
    else:
        model = FCNSG(dimension=D, num_layers=L).double()
    if GPU_IN_USE:
        model.cuda(device_id)
    # print(torch.cuda.device_count())
    print(model)
    #########################################
    cfix = "D{}_PW{}_BS{}_LR{}_CW{}_WD{}_HF{}".format(D,PAIRWISE,BATCH_SIZE,LRATE,USECLASSWT,WEIGHT_DECAY,HASHFULL)
    fix = "_MH_COMP{}_K{}_{}".format(MHCOMPUTATION,K,cfix) if MH  else "_FH_{}".format(cfix)

    print("RUNCONFIG:", fix)
    data_files = ["train.txt", "train_small_ub.txt", "test.txt", "test_small_ub.txt"]
    data_dirs = list(map(lambda f: join(cur_dir, DATASET, "data", f), data_files))
    print(data_files)
    time_file = join(cur_dir, DATASET, "record", "time_record.txt")

    if not ASYNC:
        record_files = ["acc{}_L{}.txt".format(fix, L), "val_acc{}_L{}.txt".format(fix, L),
                        "loss{}_L{}.txt".format(fix, L), "val_loss{}_L{}.txt".format(fix, L),
                        "final_prediction{}_L{}.txt".format(fix, L),
                        "checkpoint{}_L{}.ckpt".format(fix, L),
                        "index{}_L{}.txt".format(fix, L)] # TODO final preduction for async
        record_dirs = list(map(lambda f: join(cur_dir, DATASET, "record", f), record_files))
        train(data_dirs, D, model, MH, time_file, record_dirs)
    else:
        mp.set_start_method('spawn')
        model.share_memory()
        processes = []
        all_record_dirs = []
        for p_id in range(PROCESS):
            record_files = ["pid{}_acc{}_L{}.txt".format(p_id, fix, L), "pid{}_val_acc{}_L{}.txt".format(p_id, fix, L),
                            "pid{}_loss{}_L{}.txt".format(p_id, fix, L), "pid{}_val_loss{}_L{}.txt".format(p_id, fix, L)]
            record_dirs = list(map(lambda f: join(cur_dir, DATASET, "record", f), record_files))
            all_record_dirs.append(record_dirs)
            p = mp.Process(target=train, args=(data_dirs, D, model, MH),
                           kwargs={"time_file": time_file, "record_files": record_dirs, "p_id": p_id})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # File combination
        acc, valacc, loss, valloss = [], [], [], []
        for fnames in all_record_dirs:
            acc.append(np.loadtxt(fnames[0]))
            valacc.append(np.loadtxt(fnames[1]))
            loss.append(np.loadtxt(fnames[2]))
            valloss.append(np.loadtxt(fnames[3]))
        acc = np.mean(np.array(acc), axis=0).ravel()
        valacc = np.mean(np.array(valacc), axis=0).ravel()
        loss = np.mean(np.array(loss), axis=0).ravel()
        valloss = np.mean(np.array(valloss), axis=0).ravel()

        acc_name = join(cur_dir, DATASET, "record", "[ASYNC]acc{}_L{}.txt".format(fix, L))
        valacc_name = join(cur_dir, DATASET, "record", "[ASYNC]val_acc{}_L{}.txt".format(fix, L))
        loss_name = join(cur_dir, DATASET, "record", "[ASYNC]loss{}_L{}.txt".format(fix, L))
        valloss_name = join(cur_dir, DATASET, "record", "[ASYNC]val_loss{}_L{}.txt".format(fix, L))
        np.savetxt(acc_name, acc)
        np.savetxt(valacc_name, valacc)
        np.savetxt(loss_name, loss)
        np.savetxt(valloss_name, valloss)
