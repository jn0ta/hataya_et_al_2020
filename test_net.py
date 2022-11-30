import time
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import copy
from torch.utils.data.dataset import Subset
from policy import Policy
import pathlib
import torch.nn.functional as F
import sys
import numpy as np

# ================================ classes ================================
# print a string to a file
class Print_to_file:
    def __init__(self, _filename) -> None:
        self._filename = _filename

    def print(self, _str:str):
        with open(self._filename, 'a', encoding="utf-8") as _f:
            _f.write(_str)
            _f.write("\n")

# ================================ functions ================================

def temp_train ( _arg_dict:dict):
    _full_aug_trainloader = _arg_dict["dataloader"]["full_aug_train"]
    _use_cuda = _arg_dict["model"]["use_cuda"]
    _loss_fn = _arg_dict["loss_fn"]
    _test_net = _arg_dict["model"]["test_net"]
    _optimizer = _arg_dict["optimizer"]["test_net"]
    
    _test_net.train()
    _train_loss = 0.0
    _correct = 0
    _total = 0
    
    for _batch_idx, (_input, _target) in enumerate(_full_aug_trainloader):

        _input_size = _input.size()[0]
        if _use_cuda:
            _input, _target = _input.cuda(), _target.cuda()

        _output = _test_net(_input)
        _loss = _loss_fn(_output, _target)

        _train_loss += _loss.data
        _, _pred = torch.max(_output, 1)
        _correct += _pred.eq(_target).sum()
        _total += _input_size

        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
    
    return _train_loss / (_batch_idx + 1), 100*_correct/_total

def temp_test(_arg_dict:dict):
    _testloader = _arg_dict["dataloader"]["test"]
    _use_cuda = _arg_dict["model"]["use_cuda"]
    _loss_fn = _arg_dict["loss_fn"]
    _test_net = _arg_dict["model"]["test_net"]

    _test_net.eval()
    _test_loss = 0.0
    _correct = 0
    _total = 0

    for _batch_idx, (_input, _target) in enumerate(_testloader):
        _input_size = _input.size()[0]
        if _use_cuda:
            _input, _target = _input.cuda(), _target.cuda()

        _output = _test_net(_input)
        _loss = _loss_fn(_output, _target)

        _test_loss += _loss.data
        _, _pred = torch.max(_output.data, 1)
        _correct += _pred.eq(_target).sum()
        _total += _input_size

    return _test_loss / (_batch_idx + 1), 100*_correct/_total

def test_modelTraining( _arg_dict:dict):
    _ptf = _arg_dict["ptf"]["ptf"]
    _ptf2 = _arg_dict["ptf"]["ptf2"]
    _ptf.print("start TEST model training -->")
    _T1 = _arg_dict["N_iter"]["T1"]
    _test_net_optimizer = _arg_dict["optimizer"]["test_net"]
    _test_net_scheduler = _arg_dict["scheduler"]["test_net"]

    for _epoch in range(_T1):
        torch.cuda.synchronize()
        _time_one_epoch = time.time()

        _train_loss, _train_acc= temp_train( _arg_dict)
        _test_net_scheduler.step()

        torch.cuda.synchronize()
        _time_one_epoch = time.time() - _time_one_epoch
        _ptf.print(" epoch "+str(_epoch+1)+" lr:{:.3f}".format(_test_net_optimizer.param_groups[0]['lr'])+
                " train_loss:{:.5f}".format(_train_loss)+" train_acc:{:.2f}".format(_train_acc)+" processing_time:{:.3f}".format(_time_one_epoch))
        
        if (_epoch+1)%10 == 0:
            _test_loss, _test_acc = temp_test(_arg_dict)
            _ptf.print(" test_loss:{:.3f}".format(_test_loss)+" test_acc:{:.3f}".format(_test_acc))
        if _epoch == _T1-1:
            _test_loss, _test_acc = temp_test(_arg_dict)
            _ptf2.print(str(_test_loss.item())+","+str(_test_acc.item())) # record the test loss and accuracy of the LAST epoch

    _ptf.print("TEST model training is DONE")
    return

# ================================ configurations ================================

net_name = "resnet18"       #"simple_cnn"

dataset_name = "CIFAR10"
batch_size = 64             # batch size cant be 1
trainset_rate = 0.8         # = training set size / (training set size + validation set size)

policy_save_filename = 

T1 = 200        # number of epochs for model training
alpha = 0.1     # learning rate for model training

FILENAME = str("testNet_"+
                net_name+"_"+
                dataset_name+"_"+
                str(T1)+",-,-,-_"+
                str(alpha)+",-,-_"+
                "-_"+
                "-_"+
                ".txt")

today_date = datetime.datetime.today().replace(microsecond=0)

ptf = Print_to_file(FILENAME)
ptf.print("================================================================================================================================")
ptf.print(str(today_date))
ptf.print("================================================================================================================================")

FILEANME2 = "result-only_"+FILENAME # a csv-like file to record essential values(e.g., test loss, test accuracy)
ptf2 = Print_to_file(FILEANME2)
ptf2.print("================================================================================================================================")
ptf2.print(str(today_date))
ptf2.print("================================================================================================================================")

# ================================ net(=model) preparation ================================
ptf.print("models are under building ...")
net_func_dicts = {
    "resnet18" : lambda : models.resnet18(), 
    #"simple_cnn" : lambda : Simple_CNN()
}
net_func = net_func_dicts[net_name]
test_net = net_func()
ptf.print(f" model : {net_name}")

use_cuda = torch.cuda.is_available()
if use_cuda:
    ptf.print(" USING CUDA")
#    net.cuda()
    test_net.cuda()
    ptf.print(" model on the GPU")
    ptf.print(f" Num of using GPUs\t\t: {torch.cuda.device_count()}")
    #net = torch.nn.DataParallel(net)   # https://torch.classcat.com/2021/06/06/pytorch-1-8-tutorials-beginner-data-parallel/
                                        # this leads an error when saving a model https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
    cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    ptf.print(f" cudnn.benchmark\t\t: {cudnn.benchmark}")
    ptf.print(f" torch.backends.cudnn.enabled\t: {torch.backends.cudnn.enabled}")

ptf.print("model building is DONE")

# ================================ policy preparation ================================
ptf.print("policy building ...")
policy = Policy.madao_policy(temperature=0.1,
                                 mean=torch.as_tensor((0.4914, 0.4822, 0.4465)),
                                 std=torch.as_tensor((0.2023, 0.1994, 0.2010)),
                                 operation_count=2)

policy.load_state_dict( torch.load(policy_save_filename) )

if use_cuda: 
    policy.cuda()
    ptf.print(" augmentation policy(nn.Module) on the GPU")

ptf.print("policy building is DONE")

# ================================ data preparatioin ================================
transform_train = transforms.Compose([
        policy.pil_forward, # augmentation, by Hataya et al., 2022
        transforms.ToTensor()
    ])

transform_test = transforms.Compose([ transforms.ToTensor() ])


if dataset_name == "CIFAR10":
    full_aug_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif dataset_name == "MNIST":
    full_aug_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

full_aug_trainloader = torch.utils.data.DataLoader(full_aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

ptf.print("datasets and dataloaders HAVE BEEN prepared")
ptf.print(f" dataset name : {dataset_name}")
ptf.print(f" batch size : {batch_size}")
ptf.print(f" len(full_aug_trainset) : {len(full_aug_trainset)}, len(testset) : {len(testset)}") 

# ================================ optimizers, loss function, scheduler preparetion ================================

loss_fn = nn.CrossEntropyLoss() # train_loss, val_loss, test_loss

test_net_optimizer = optim.SGD(test_net.parameters(), lr=alpha)
test_net_scheduler = optim.lr_scheduler.MultiStepLR(test_net_optimizer, milestones=[60, 120, 160], gamma=0.2)

# ================================ MAIN ================================

arg_dict = {
    "ptf"       : {"ptf" : ptf, "ptf2" : ptf2},
    "dataset"   : { "batch_size": batch_size, "test" : testset},
    "dataloader": { "full_aug_train" : full_aug_trainloader, "test" : testloader},
    "model"     : { "use_cuda" : use_cuda, "net_func" : net_func, "test_net" : test_net},
    "policy"    : policy,
    "lr"        : { "model_train" : alpha},
    "optimizer" : {"test_net" : test_net_optimizer},
    "scheduler" : {"test_net" : test_net_scheduler},
    "loss_fn"   : loss_fn,
    "N_iter"    : { "T1" : T1}
}

test_modelTraining(arg_dict)