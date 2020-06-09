import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from .a2c_ppo_acktr.utils import init

# from a2c_ppo_acktr.baye_layer_new import BayesianConv2D, BayesianLinear
from .baye_layer import BayesianLinear

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, ratio_init=32., base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.taskcla = taskcla
        self.num_inputs = obs_shape
        self.base = base(obs_shape[0], taskcla, 1/ratio_init, **base_kwargs)
        
        self.dist = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.dist.append(DiagGaussian(self.base.output_size, n))

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, task_num):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, task_num, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_num, sample=True)
        dist = self.dist[task_num](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, task_num):
        value, _, _ = self.base(inputs, rnn_hxs, masks, task_num)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, task_num, sample=False, sample_num=1):
        
        BATCH_SIZE=len(inputs)
        sample_value = torch.zeros(sample_num, BATCH_SIZE, 1)
        sample_action_log_probs = torch.zeros(sample_num, BATCH_SIZE, 1)
        sample_dist_entropy = torch.zeros(sample_num)
        
        for i in range(sample_num):

            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, task_num, sample=sample)
            dist = self.dist[task_num](actor_features)

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()
            
            sample_value[i] = value[task_num]
            
            sample_action_log_probs[i] = action_log_probs
            sample_dist_entropy[i] = dist_entropy

        return sample_value, sample_action_log_probs.mean(0), sample_dist_entropy.mean(0), rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, taskcla, ratio_init= 1/16, recurrent=False, hidden_size=16):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.taskcla = taskcla
        
        print ('hidden size : ', hidden_size)
            
        self.nn1_actor = BayesianLinear(num_inputs, hidden_size, ratio= ratio_init)
        self.nn2_actor = BayesianLinear(hidden_size, hidden_size, ratio= ratio_init)

        self.nn1_critic = BayesianLinear(num_inputs, hidden_size, ratio= ratio_init)
        self.nn2_critic = BayesianLinear(hidden_size, hidden_size, ratio= ratio_init)

        self.critic_linear = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.critic_linear.append(init_(torch.nn.Linear(hidden_size,1)))

#         self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, task_num, sample = False):
        x_ciritic = inputs
        x_actor = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
            
        x_ciritic = F.tanh(self.nn1_critic(x_ciritic, sample))
        hidden_critic = F.tanh(self.nn2_critic(x_ciritic, sample))
        
        x_actor = F.tanh(self.nn1_actor(x_actor, sample))
        hidden_actor = F.tanh(self.nn2_actor(x_actor, sample))
        critic_output=[]
        for t,i in self.taskcla:
            critic_output.append((self.critic_linear[t](hidden_critic)))

        return critic_output[task_num], hidden_actor, rnn_hxs



