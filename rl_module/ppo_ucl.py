import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from copy import deepcopy

from .ucl_model import MLPBase
from .baye_layer import BayesianLinear

class PPO_UCL():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 conv_net=False,
                 beta = 0.03,
                 rho_init = -2.783,
                ):

        self.actor_critic = actor_critic
        self.old_actor_critic = deepcopy(self.actor_critic)
        
        self.conv_net = conv_net

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
        self.num_mini_batch = num_mini_batch
        self.saved = 0
        self.beta = beta
        self.rho = rho_init
        
        print ('beta : ', self.beta)
        print ('rho : ', self.rho)

    def update(self, rollouts, task_num, sample_flag=True, sample_num=1):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num,sample=sample_flag, sample_num=sample_num)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                
                reg_loss = self.custom_regularization(self.num_mini_batch)
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + reg_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    
    def update_saved(self):

        self.saved = 1
        return
    
    def update_old_actor_critic(self):
        
        self.old_actor_critic = deepcopy(self.actor_critic)
    
    def custom_regularization(self, mini_batch_size):

        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0

        if self.conv_net:
            prev_rho = nn.Parameter(torch.Tensor(self.actor_critic.num_inputs).uniform_(1, 1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho))

            if isinstance(self.old_actor_critic.base, CNNBase) == False or isinstance(self.actor_critic.base,
                                                                                      CNNBase) == False:
                return

        else:
            prev_rho = nn.Parameter(torch.Tensor(self.actor_critic.num_inputs).uniform_(1, 1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho)).cuda()
            if isinstance(self.old_actor_critic.base, MLPBase) == False or isinstance(self.actor_critic.base,
                                                                                      MLPBase) == False:
                return

        for (module_name, saver_module), (_, trainer_module) in zip(self.old_actor_critic.named_children(),
                                                                    self.actor_critic.named_children()):
            for (layer_name, saver_layer), (_, trainer_layer) in zip(saver_module.named_children(),
                                                                     trainer_module.named_children()):

                if not (((layer_name == 'critic_linear') or (module_name == 'dist'))):

                    # calculate mu regularization

                    trainer_weight_mu = trainer_layer.weight_mu
                    saver_weight_mu = saver_layer.weight_mu
                    trainer_bias_mu = trainer_layer.bias_mu
                    saver_bias_mu = saver_layer.bias_mu

                    trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
                    saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
                    trainer_bias_sigma = torch.log1p(torch.exp(trainer_layer.bias_rho))
                    saver_bias_sigma = torch.log1p(torch.exp(saver_layer.bias_rho))

                    out_features, in_features = saver_weight_mu.shape
                    curr_sigma = saver_weight_sigma.expand(out_features,in_features)
                    prev_sigma = prev_weight_sigma.expand(out_features,in_features)

                    L1_sigma = saver_weight_sigma
                    L2_sigma = torch.min(curr_sigma, prev_sigma)
                    prev_weight_sigma = saver_weight_sigma

                    mu_weight_reg = (torch.div(trainer_weight_mu-saver_weight_mu, L2_sigma)).norm(2)**2
                    mu_bias_reg = (torch.div(trainer_bias_mu-saver_bias_mu, saver_bias_sigma)).norm(2)**2

                    L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
                    L1_mu_bias_reg = (torch.div(saver_bias_mu**2,saver_bias_sigma**2)*(trainer_bias_mu - saver_bias_mu)).norm(1)

                    std_init = np.log(1+np.exp(self.rho))

                    mu_weight_reg = mu_weight_reg * (std_init ** 2)
                    mu_bias_reg = mu_bias_reg * (std_init ** 2)
                    L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
                    L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)

                    weight_sigma = trainer_weight_sigma**2 / saver_weight_sigma**2
                    bias_sigma = trainer_bias_sigma**2 / saver_bias_sigma**2

                    normal_weight_sigma = trainer_weight_sigma**2
                    normal_bias_sigma = trainer_bias_sigma**2

                    sigma_weight_reg_sum += (weight_sigma - torch.log(weight_sigma)).sum()
                    sigma_weight_reg_sum += (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()
                    sigma_bias_reg_sum += (bias_sigma - torch.log(bias_sigma)).sum() 
                    sigma_bias_reg_sum += (normal_bias_sigma - torch.log(normal_bias_sigma)).sum()

                    mu_weight_reg_sum += mu_weight_reg
                    mu_bias_reg_sum += mu_bias_reg
                    L1_mu_weight_reg_sum += L1_mu_weight_reg
                    L1_mu_bias_reg_sum += L1_mu_bias_reg

        # L2 loss
        loss =  (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * mini_batch_size)
        # L1 loss
        loss = loss + self.saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.beta * (sigma_weight_reg_sum + sigma_bias_reg_sum) / (2 * mini_batch_size)

        return loss