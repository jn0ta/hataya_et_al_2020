import time, datetime, os, pathlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from policy import Policy
import matplotlib.pyplot as plt

# ================================ classes ================================
# print a string to a file
class Print_to_file:
    def __init__(self, _filename:str) -> None:
        #self._filename = _filename
        self.jstTime = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).replace(microsecond=0)
        self.path = pathlib.Path('program_output/'+str(self.jstTime.date()))
        if not self.path.exists(): self.path.mkdir(parents=True)
        self.path = pathlib.Path('program_output/'+str(self.jstTime.date())+"/"+_filename)
        self.print("================================================================================================================================")
        self.print(str(self.jstTime).replace('+09:00', ''))
        self.print("================================================================================================================================")

    def print(self, _str:str):
        #with open(self._filename, 'a', encoding="utf-8") as _f:
        with open(self.path, 'a', encoding="utf-8") as _f:
            _f.write(_str)
            _f.write("\n")

# simple cnn
class Simple_CNN(nn.Module): # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ================================ functions ================================

def train ( _arg_dict:dict):
    _trainloader = _arg_dict["dataloader"]["train"]
    _use_cuda = _arg_dict["model"]["use_cuda"]
    _loss_fn = _arg_dict["loss_fn"]
    _net = _arg_dict["model"]["net"]
    _optimizer = _arg_dict["optimizer"]["net"]
    
    _net.train()
    _train_loss = 0.0
    _correct = 0
    _total = 0
    
    for _batch_idx, (_input, _target) in enumerate(_trainloader):

        _input_size = _input.size()[0]
        if _use_cuda:
            _input, _target = _input.cuda(), _target.cuda()

        _output = _net(_input)
        _loss = _loss_fn(_output, _target)

        _train_loss += _loss.data
        _, _pred = torch.max(_output, 1)
        _correct += _pred.eq(_target).sum()
        _total += _input_size

        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
    
    return _train_loss / (_batch_idx + 1), 100*_correct/_total

def test(_arg_dict:dict):
    _testloader = _arg_dict["dataloader"]["test"]
    _use_cuda = _arg_dict["model"]["use_cuda"]
    _loss_fn = _arg_dict["loss_fn"]
    _net = _arg_dict["model"]["net"]

    _net.eval()
    _test_loss = 0.0
    _correct = 0
    _total = 0

    for _batch_idx, (_input, _target) in enumerate(_testloader):
        _input_size = _input.size()[0]
        if _use_cuda:
            _input, _target = _input.cuda(), _target.cuda()

        _output = _net(_input)
        _loss = _loss_fn(_output, _target)

        _test_loss += _loss.data
        _, _pred = torch.max(_output.data, 1)
        _correct += _pred.eq(_target).sum()
        _total += _input_size

    return _test_loss / (_batch_idx + 1), 100*_correct/_total

def modelTraining( _arg_dict:dict):
    _ptf = _arg_dict["ptf"]["ptf"]
    _ptf2 = _arg_dict["ptf"]["ptf2"]
    _ptf.print("start TEST model training -->")
    _T0 = _arg_dict["N_iter"]["T0"]
    _net_optimizer = _arg_dict["optimizer"]["net"]
    _net_scheduler = _arg_dict["scheduler"]["net"]

    for _epoch in range(_T0):
        torch.cuda.synchronize()
        _time_one_epoch = time.time()

        _train_loss, _train_acc= train( _arg_dict)
        _net_scheduler.step()

        torch.cuda.synchronize()
        _time_one_epoch = time.time() - _time_one_epoch
        _ptf.print(" epoch "+str(_epoch+1)+" lr:{:.3f}".format(_net_optimizer.param_groups[0]['lr'])+
                " train_loss:{:.5f}".format(_train_loss)+" train_acc:{:.2f}".format(_train_acc)+" processing_time:{:.3f}".format(_time_one_epoch))
        
        if (_epoch+1)%10 == 0:
            _test_loss, _test_acc = test(_arg_dict)
            _ptf.print(" test_loss:{:.3f}".format(_test_loss)+" test_acc:{:.3f}".format(_test_acc))
        if _epoch == _T0-1:
            _test_loss, _test_acc = test(_arg_dict)
            _ptf2.print(str(_test_loss.item())+","+str(_test_acc.item())) # record the test loss and accuracy of the LAST epoch

    _ptf.print("TEST model training is DONE")
    return

# ================================ configurations ================================

net_name = "simple_cnn"

dataset_name = "CIFAR10"
batch_size = 64             # batch size cant be 1
val_test_batch_size = batch_size * 2                                # ACCORDING TO hataya et.al., 2020, set this
mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))     # mean and starndard deviation for the transforms : Normalize()

policy_save_filename = "polDict_2022-12-27-9-10-3_simple_cnn_cifar10_60_20_2500.pt"
                        
T0 = 100        # number of epochs for model training
milestones = [int(T0*3/10), int(T0*6/10), int(T0*8/10)] # the milestones for optim.lr_scheduler.MultiStepLR()
alpha = 0.1     # learning rate for model training

use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
else:
    cuda_visible_devices = "None"
FILENAME = str("testNet_"+
                net_name+"_"+
                dataset_name+"_"+
                str(T0)+",-,-,-_"+
                str(alpha)+",-,-_"+
                "-_"+
                "-_"+
                "_"+cuda_visible_devices+
                ".txt")
FILEANME2 = "result-only_"+FILENAME # a csv-like file to record essential values(e.g., test loss, test accuracy)
ptf = Print_to_file(FILENAME)
ptf2 = Print_to_file(FILEANME2)

# ================================ net(=model) preparation ================================
ptf.print("models are under building ...")
net_func_dicts = {
    "resnet18" : lambda : models.resnet18(), 
    "simple_cnn" : lambda : Simple_CNN()
}
net_func = net_func_dicts[net_name]
net = net_func()
ptf.print(f" model : {net_name}")

if use_cuda:
    ptf.print(" USING CUDA")
    net.cuda()
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
                                 mean=torch.as_tensor(mean_std[0]),
                                 std=torch.as_tensor(mean_std[1]),
                                 operation_count=2)

policy.load_state_dict( torch.load(policy_save_filename) )
ptf.print(f" the loaded file : {policy_save_filename}")

if use_cuda: 
    policy.cuda()
    ptf.print(" augmentation policy(nn.Module) on the GPU")

ptf.print("policy building is DONE")

# debug
#for pp in policy.parameters():
#    print(pp)
#sys.exit("debug")

# ================================ data preparatioin ================================

transform_aug_train = transforms.Compose([
    policy.pil_forward, # augmentation, by Hataya et al., 2022
    transforms.ToTensor(),
    transforms.Normalize(mean_std[0],mean_std[1])
])

transform_test = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean_std[0],mean_std[1])
])

if dataset_name == "CIFAR10":
    augTrainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif dataset_name == "MNIST":
    augTrainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_aug_train)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

augTrainloader = torch.utils.data.DataLoader(augTrainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=val_test_batch_size, shuffle=False, num_workers=8)

ptf.print("datasets and dataloaders HAVE BEEN prepared")
ptf.print(f" dataset name : {dataset_name}")
ptf.print(f" batch size : {batch_size}")
ptf.print(f" len(augTrainset) : {len(augTrainset)}, len(testset) : {len(testset)}") 

# ================================ optimizers, loss function, scheduler preparetion ================================

loss_fn = nn.CrossEntropyLoss() # train_loss, test_loss

net_optimizer = optim.SGD(net.parameters(), lr=alpha)
net_scheduler = optim.lr_scheduler.MultiStepLR(net_optimizer, milestones=milestones, gamma=0.2)

# ================================ MAIN ================================

arg_dict = {
    "ptf"       : {"ptf" : ptf, "ptf2" : ptf2},
    "dataset"   : { "batch_size": batch_size, "test" : testset},
    "dataloader": { "train" : augTrainloader, "test" : testloader},
    "model"     : { "use_cuda" : use_cuda, "net_func" : net_func, "net" : net},
    "policy"    : policy,
    "lr"        : { "model_train" : alpha},
    "optimizer" : {"net" : net_optimizer},
    "scheduler" : {"net" : net_scheduler},
    "loss_fn"   : loss_fn,
    "N_iter"    : { "T0" : T0}
}

modelTraining(arg_dict)