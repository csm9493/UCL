import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args

args = get_args()

from bayes_layer import BayesianLinear, BayesianConv2D, _calculate_fan_in_and_fan_out

class Appr(object):
    

    def __init__(self, model, nepochs=100, sbatch=256, lr=0.001, 
                 lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None, split=False):

        self.model = model
        self.model_old = deepcopy(self.model)
        
        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_rho = args.lr_rho
        self.lr_min = lr / (lr_factor ** 5)
        self.lr_factor = lr_factor
        self.lr_patience = 5
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.epoch = 0
        self.saved = 0
        self.split = split
        self.beta = args.beta
        
        self.drop = [20,40,60,75,90]
        
        self.param_name = []
        
        for (name, p) in self.model.named_parameters():
            self.param_name.append(name)
        
        self.optimizer = self._get_optimizer()
        
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(para ,ms[0])

        return

    def _get_optimizer(self, lr=None, lr_rho = None):
        if lr is None: lr = self.lr
        if lr_rho is None: lr_rho = self.lr_rho
        if args.optimizer == 'Adam':
            return Adam(self.model.parameters(), lr=lr, lr_rho=lr_rho, param_name = self.param_name)
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)
    
    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        lr_rho = self.lr_rho
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr, lr_rho)
        
        # Loop epochs
        for e in range(self.nepochs):
            self.epoch = self.epoch + 1
            # Train
            clock0 = time.time()

            num_batch = xtrain.size(0)
            
            self.train_epoch(t, xtrain, ytrain)
            
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, xtrain, ytrain)
            
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / num_batch,
                1000 * self.sbatch * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t * self.nepochs) + e, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t):
                xvalid_t=data[task]['valid']['x'].cuda()
                yvalid_t=data[task]['valid']['y'].cuda()
                    
                valid_loss_t, valid_acc_t = self.eval(task, xvalid_t, yvalid_t)
                self.logger.add(epoch=(t * self.nepochs) + e, task_num=task + 1, valid_loss=valid_loss_t,
                                valid_acc=valid_acc_t)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    lr_rho /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr, lr_rho)
            print()

            utils.freeze_model(self.model_old)  # Freeze the weights
            
            
        # Restore best
        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        self.saved = 1

        self.logger.save()

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()
        
        # Loop batches
        
        for i in range(0, len(r), self.sbatch):
            self.iteration += 1
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]

            # Forward current model
            mini_batch_size = len(targets)
            
            if self.split:
                output = F.log_softmax(self.model(images, sample=True)[t],dim=1)
            else:
                output = self.model(images, sample=True)
            loss = F.nll_loss(output, targets, reduction='sum')
            loss = self.custom_regularization(self.model_old, self.model, mini_batch_size, loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()
        
        # Loop batches
        with torch.no_grad():
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r):
                    b = r[i:i + self.sbatch]
                else:
                    b = r[i:]
                images = x[b]
                targets = y[b]
                
                # Forward
                mini_batch_size = len(targets)
                if self.split:
                    output = F.log_softmax(self.model(images, sample=False)[t],dim=1)
                else:
                    output = self.model(images, sample=False)
                loss = F.nll_loss(output, targets, reduction='sum')
                
                _, pred = output.max(1)
                hits = (pred == targets).float()
                
                total_loss += loss.data.cpu().numpy()
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(b)



        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg

# custom regularization

    def custom_regularization(self, saver_net, trainer_net, mini_batch_size, loss=None):
        
        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        sigma_weight_normal_reg_sum = 0
        sigma_bias_normal_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0
        
        out_features_max = 512
        alpha = args.alpha
        if self.saved:
            alpha = 1
        
        if args.conv_net:
            if args.experiment == 'omniglot':
                prev_weight_strength = nn.Parameter(torch.Tensor(1,1,1,1).uniform_(0,0))
            else:
                prev_weight_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform_(0,0))

        else:
            prev_weight_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(0,0))
        
        for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
            if isinstance(trainer_layer, BayesianLinear)==False and isinstance(trainer_layer, BayesianConv2D)==False:
                continue
            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias = trainer_layer.bias
            saver_bias = saver_layer.bias
            
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            
            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            
            if isinstance(trainer_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * args.ratio)
            if isinstance(trainer_layer, BayesianConv2D):
                std_init = math.sqrt((2 / fan_out) * args.ratio)
            
            saver_weight_strength = (std_init / saver_weight_sigma)

            if len(saver_weight_mu.shape) == 4:
                out_features, in_features, _, _ = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features,1,1)
                prev_strength = prev_weight_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features)
                if len(prev_weight_strength.shape) == 4:
                    feature_size = in_features // (prev_weight_strength.shape[0])
                    prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0],-1)
                    prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                    prev_weight_strength = prev_weight_strength.reshape(-1,1)
                prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
            
            L2_strength = torch.max(curr_strength, prev_strength)
            bias_strength = torch.squeeze(saver_weight_strength)
            
            L1_sigma = saver_weight_sigma
            bias_sigma = torch.squeeze(saver_weight_sigma)
            
            prev_weight_strength = saver_weight_strength
            
            mu_weight_reg = (L2_strength * (trainer_weight_mu-saver_weight_mu)).norm(2)**2
            mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2
            
            L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)
            
            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
            
            weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
            
            normal_weight_sigma = trainer_weight_sigma**2
            
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_normal_reg_sum = sigma_weight_normal_reg_sum + (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()
            
            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg
            
        # elbo loss
        loss = loss / mini_batch_size
        # L2 loss
        loss = loss + alpha * (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * mini_batch_size)
        # L1 loss
        loss = loss + self.saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.beta * (sigma_weight_reg_sum + sigma_weight_normal_reg_sum) / (2 * mini_batch_size)
            
        return loss

