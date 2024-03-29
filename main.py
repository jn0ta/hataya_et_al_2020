from __future__ import annotations

import warnings
from typing import Iterable

import chika
import homura
import torch
from policy import Policy
from homura import trainers
from homura.metrics import accuracy
from torch import autograd
from torch.autograd import grad
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY
from data import get_data, Data

import torchvision.models as models
import sys
import datetime

# jun ota addition
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import pathlib
from savePolicy import *

def param_to_vector(p: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    # unlike pytorch's parameters_to_vec,
    # no device check, but allows non-contiguous params, which may happens in second derivatives
    return torch.cat([param.reshape(-1) for param in p])


class NeumannTrainer(trainers.SupervisedTrainer):
    def __init__(self, *args, **kwargs):
        self.policy: torch.nn.Module = kwargs.pop('policy')
        self.cfg = kwargs.pop('cfg')

        super().__init__(*args, **kwargs)
        self.policy.to(self.device)
        #self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

        # jun ota edition
        #self.policy_optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.cfg.lr)
        self.policy_optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.cfg.lr, alpha=0.9)
        #self.N_polIters = 0

    def infer_batch_size(self,
                         data
                         ) -> int:
        if self.is_train:
            return data.train_da.size(0)
        else:
            return super().infer_batch_size(data)

    def data_preprocess(self,
                        data: Data | tuple[torch.Tensor, torch.Tensor]
                        ):
        if self.is_train:
            return data.to(self.device, non_blocking=self._cuda_nonblocking)
        else:
            return super().data_preprocess(data)

    def set_aug_grad(self,
                     grads: Iterable) -> None:
        for grad, param in zip(grads, self.policy.parameters()):
    #        param.grad = grad
            # jun ota edition
            if grad is not None:
                param.grad = - grad
            else:
                param.grad = grad

    @torch.no_grad()
    def approx_ihvp(self,
                    in_grad: torch.Tensor,
                    out_grad: torch.Tensor,
                    params) -> torch.Tensor:
        approx = out_grad.detach().clone()
        temp = approx
        for _ in range(self.cfg.approx_iters):
            with torch.enable_grad():
                # hessian vector product
                out_grad = param_to_vector(grad(in_grad, params, grad_outputs=temp, retain_graph=True))
    #        temp -= self.cfg.lr * out_grad.clone()
            temp -= self.cfg.alpha * out_grad.clone() # jun ota edition
            approx += temp
    #    return self.cfg.lr * approx
        return approx # jun ota edition

    def iteration(self,
                  data: dict or tuple) -> None:
        if not self.is_train:
            self.model.eval()
            input, target = data
            output = self.model(input)
            loss = self.loss_f(output, target)
            self.reporter.add("loss", loss)
            self.reporter.add("accuracy", accuracy(output, target))
            return

        self.model.train()
        self.policy.train()
        data: Data
        is_warmup = self.cfg.warmup_epochs > self.epoch
        self.optimizer.zero_grad()
        if is_warmup or self.step == 0 or self.step % self.cfg.da_interval != 0:
            in_output = self.model(data.train_da)
            in_loss = self.loss_f(in_output, data.train_labels_da)
            in_loss.backward()
            self.optimizer.step()
            self.reporter.add("loss", in_loss.detach_())
            self.reporter.add("accuracy", accuracy(in_output, data.train_labels_da))
            
            # jun ota edition, check policy.pil_forward output
            #if self.step in [0,1,2,3,4]:
                #print(f" type(data.train_da) : {type(data.train_da)}")     # <-- <class 'torch.Tensor'>
            #    plt.imshow(transforms.ToPILImage()(data.train_da[0])); plt.savefig("pil_forward_check_"+str(self.step)+".png")
            
            return

        # jun ota debug
        #print(f"when update the policy, self.step : {self.step}")

        # outer step
        # final inner step
        self.optimizer.zero_grad()
        input = self.policy(data.train_no_da)

        # jun ota edition
        #plt.imshow(transforms.ToPILImage()(data.train_no_da[0])); plt.savefig("policy_check__Bfr_"+str(self.step)+".png")
        #plt.imshow(transforms.ToPILImage()(input[0])); plt.savefig("policy_check__Aft_"+str(self.step)+".png")
        #_ = input("paused, hit Enter to continue")

        out_output = self.model(input)
        in_loss = self.loss_f(out_output, data.train_labels_no_da)
        in_loss.backward(retain_graph=True)
        self.optimizer.step()

        # actual outer step
        self.model.eval()
        # DistributedDataParallel does not support autograd.grad
        val_output = self.model(data.val)
        out_loss = self.loss_f(val_output, data.val_labels)
        self.policy_update(in_loss, out_loss)
        self.optimizer.zero_grad()

        #self.N_polIters += 1;print("self.N_polIters:",self.N_polIters) # jun ota edition

        self.reporter.add("loss/val", out_loss.detach_())

    def policy_update(self,
                      in_loss: torch.Tensor,
                      out_loss: torch.Tensor
                      ) -> None:
        model_params = list(self.model.parameters())
        # ∂f/∂θ
        in_grad = param_to_vector(autograd.grad(in_loss, model_params, create_graph=True))
        # ∂g/∂θ
        out_grad = param_to_vector(autograd.grad(out_loss, model_params))
        # ∂g/∂θ H^{-1}
        out_in_hessian = self.approx_ihvp(in_grad, out_grad, model_params)
        # policy is stochastic, so some parameters might not be used (i.e., None)
        # (∂g/∂θ H^{-1}) (∂/∂φ)(∂f/∂θ)
        mix_grad = autograd.grad(in_grad, self.policy.parameters(), grad_outputs=out_in_hessian, allow_unused=True)
        self.set_aug_grad(mix_grad)

        # jun ota addition
        #if self.epoch == self.cfg.warmup_epochs:
        #    print("  policy gradient")
        #    for pp in self.policy.parameters():
        #        print(pp.grad.data)

        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()


@chika.config
class OptimConfig:
    lr: float = 0.1
    wd: float = 5e-4 #0.0 # 5e-4 # jun ota edition
    epochs: int = 300
    no_nesterov: bool = True

    def __post_init__(self):
        self.nesterov = not self.no_nesterov


@chika.config
class DatasetConfig:
    name: str = chika.choices("cifar10")   # chika.choices("cifar10", "cifar100", "svhn", ) # jun ota edition
    batch_size: int = 128   # jun ota edition
    download: bool = True   # False # jun edition
    train_size: int = None
    val_size: int = 5000   # 4_000 # jun ota edition
    no_pin_memory: bool = False
    num_workers: int = 0    # 1     # jun ota edition, num_workers MUST be 0, otherwise the execution is struggling

    def __post_init__(self):
        self.da_interval = None
        self.pin_memory = not self.no_pin_memory


@chika.config
class MetaConfig:
    lr: float = 0.01            # 1e-3 # jun ota edition
    da_interval: int = 60       #625* 4  # <-- 40000/64 = 625     312 # 60 # jun ota edition
    warmup_epochs: int = 30     #170 # 30 # jun ota edition
    approx_iters: int = 5
    temperature: float = 0.1
    # jun ota edition
    alpha: float = 0.001


@chika.config
class Config:
    data: DatasetConfig
    optim: OptimConfig
    meta: MetaConfig

    model_name: str = chika.choices("wrn28_10")  # chika.choices("wrn28_2", "wrn40_2") # jun ota edition
    seed: int = None        # 1     # jun ota edition, setting None means RANDOM (if seed = 1, NO randomness)
    gpu: int = 0            # None  # jun ota edition
    debug: bool = False
    baseline: bool = False

    def __post_init__(self):
        self.data.da_interval = self.meta.da_interval


def _main(cfg: Config):
    # jun ota edition
    todays_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).replace(microsecond=0)
    print(f"EXECUTED AT : {str(todays_date).replace('+09:00', '')}") 

    train_loader, test_loader, num_classes = get_data(cfg.data)
    model = MODEL_REGISTRY(cfg.model_name)(num_classes=num_classes)
    optimizer = homura.optim.SGD(lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.wd, multi_tensor=True,
                                 nesterov=cfg.optim.nesterov)

    # jun ota edition
    #optimizer = homura.optim.SGD(lr=cfg.optim.lr, momentum=0.0, weight_decay=cfg.optim.wd, multi_tensor=True, nesterov=cfg.optim.nesterov)
    # jun ota memo
    # the option, multi_tensor, looks just for faster training    https://github.com/huggingface/transformers/issues/9965

    scheduler = homura.lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, 5, 1e-6)
    
    # jun ota edition
    #milestones = [int(cfg.optim.epochs*3/10), int(cfg.optim.epochs*6/10), int(cfg.optim.epochs*8/10)] # the milestones for optim.lr_scheduler.MultiStepLR()
    #print(f"milestones : {milestones}") # <-- milestones : [60, 120, 160] when cfg.optim.epochs == 200
    #scheduler = homura.lr_scheduler.MultiStepLR(milestones=milestones, gamma=0.2)
    
    policy = Policy.madao_policy(temperature=cfg.meta.temperature,
                                 mean=torch.as_tensor(train_loader.mean_std[0]),
                                 std=torch.as_tensor(train_loader.mean_std[1]),
                                 operation_count=2)
    train_loader.register_policy(policy)
    with NeumannTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                        reporters=homura.reporters.TensorboardReporter("."),
                        quiet=True, # jun ota edition
                        policy=policy,
                        cfg=cfg.meta,
                        loss_val_f=F.cross_entropy) as trainer:
        trainer.scheduler.verbose = False
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
        
    # jun ota edition, saving the policy to a pytorch file xxxx.pt
    policy_save_filename = str("polDict_"+
                            str(todays_date.date())+"-"+
                            str(todays_date.hour)+"-"+str(todays_date.minute)+"-"+str(todays_date.second)+"_"+
                            cfg.model_name+"_"+cfg.data.name+"_"+
                            str(cfg.optim.epochs)+"_"+str(cfg.meta.warmup_epochs)+"_"+str(cfg.meta.da_interval)+
                            ".pt")
    policy.cpu()
    savePolicy(policy, policy_save_filename, "../../polDicts/")

    # jun ota debug
    print("---- ---- ---- ---- the below : the optimized policy parameters ---- ---- ---- ----")
    for pp in policy.parameters():
        print(pp)

@chika.main(cfg_cls=Config, change_job_dir=True)
def main(cfg):
    with homura.set_seed(cfg.seed):
        torch.cuda.set_device(cfg.gpu) # jun ota memo, this is DISCOURAGED,  see https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html
        _main(cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "The align_corners")
    main()
