import sys,time,os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
args = get_args()

if args.conv_net:
    from networks.conv_net import Net
else:
    from networks.mlp import Net

class Appr():
    """ Class implementing the Synaptic intelligence approach described in https://arxiv.org/abs/1703.04200 """

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None, log_name=None, split=False):
        super().__init__()
        self.model=model
        self.model_old=model

        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.split = split

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.c=args.c 
        self.epsilon=0.01
        if args.experiment == 'split_notmnist':
            self.epsilon = 0.001
        self.omega = {}
        self.W = {}
        self.p_old = {}
        
        n=0
        
        # Register starting param-values (needed for “intelligent synapses”).
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
        
        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        self.W = {}
        self.p_old = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.p_old[n] = p.data.clone()
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            num_batch = xtrain.size(0)
            
            self.train_epoch(t,xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            #save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t*self.nepochs)+e, task_num=t+1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t): 
                xvalid_t=data[task]['valid']['x'].cuda()
                yvalid_t=data[task]['valid']['y'].cuda()
                
                valid_loss_t,valid_acc_t=self.eval(task,xvalid_t,yvalid_t)
                self.logger.add(epoch=(t*self.nepochs)+e, task_num=task+1, valid_loss=valid_loss_t, valid_acc=valid_acc_t)
            
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        if args.conv_net:
                            pass
                            
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()
        self.update_omega(self.W, self.epsilon)
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old) # Freeze the weights
        
        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward current model
            if self.split:
                output = self.model.forward(images)[t]
            else:
                output = self.model.forward(images)
            loss=self.criterion(t,output,targets)
            
            n = 0
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()
            for n, p in self.model.named_parameters():
               if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.p_old[n] = p.detach().clone()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]
            
            # Forward
            if self.split:
                output = self.model.forward(images)[t]
            else:
                output = self.model.forward(images)
                
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
#             total_loss+=loss.data.cpu().numpy()[0]*len(b)
#             total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t>0:
            loss_reg=self.surrogate_loss()
        
        return self.ce(output,targets)+self.c*loss_reg
    
    def update_omega(self, W, epsilon):
        """After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)"""

        # Loop over all parameters
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.model.register_buffer('{}_SI_omega'.format(n), omega_new)
    
    def surrogate_loss(self):
        """Calculate SI’s surrogate loss"""
        try:
            losses = []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.model, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.model, '{}_SI_omega'.format(n))
                    # Calculate SI’s surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return 0.


