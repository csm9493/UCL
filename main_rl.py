import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import torch
from arguments_rl import get_args

from collections import deque
from rl_module.a2c_ppo_acktr.envs import make_vec_envs
from rl_module.a2c_ppo_acktr.storage import RolloutStorage
from rl_module.train_ppo import train_ppo

tstart = time.time()

# Arguments
args = get_args()
conv_experiment = [
    'roboschool',
]

# Split
##########################################################################################################################33
if args.approach == 'fine-tuning':
    log_name = '{}_{}_{}_{}'.format(args.date, args.experiment, args.approach,args.seed)
elif args.approach == 'ewc' in args.approach:
    log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.ewc_lambda)
elif args.approach == 'ucl':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}'.format(args.date, args.experiment, args.approach,args.seed, args.ucl_rho, args.ucl_beta)

if args.experiment in conv_experiment:
    log_name = log_name + '_conv'
    
"""
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
"""
########################################################################################################################
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from rl_module.ppo_model import Policy

# Inits
print('Inits...')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if args.cuda else "cpu")

if args.experiment == 'roboschool':

    obs_shape = (44,)
    
    taskcla = [(0,17),(1,17),(2,17),(3,8),(4,6),(5,6),(6,3),(7,3)]
    task_sequences = [(0,'RoboschoolHumanoid-v1'),(1,'RoboschoolHumanoidFlagrun-v1'),(2,'RoboschoolHumanoidFlagrunHarder-v1'),(3,'RoboschoolAnt-v1'),
                     (4,'RoboschoolHalfCheetah-v1'), (5,'RoboschoolWalker2d-v1'),(6,'RoboschoolHopper-v1'),(7,'RoboschoolReacher-v1')]

actor_critic = Policy(obs_shape,taskcla,).to(device)

# Args -- Approach
if args.approach == 'fine-tuning':
    from rl_module.ppo import PPO as approach
    
    agent = approach(actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.optimizer,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True)
    
elif args.approach == 'ewc':
    from rl_module.ppo_ewc import PPO_EWC as approach
    
    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        ewc_lambda= args.ewc_lambda,
        online = args.ewc_online)
    
elif args.approach == 'ucl':
    from rl_module.ppo_ucl import PPO_UCL  as approach
    
    agent = approach(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True,
        conv_net=False,
        beta = args.ucl_beta,
        rho_init=args.ucl_rho)
    

########################################################################################################################
    
tr_reward_arr = []
te_reward_arr = {}

for _type in (['mean', 'max', 'min']):
    te_reward_arr[_type] = {}
    for idx in range(len(taskcla)):
        te_reward_arr[_type]['task' + str(idx)] = []

for task_idx,env_name in task_sequences:
    
    envs = make_vec_envs(env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    obs = envs.reset()
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  obs_shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)
    
    if args.experiment == 'roboschool':
        obs_shape_real = envs.observation_space.shape
        
        
        new_obs = torch.zeros(args.num_processes, *obs_shape)
        new_obs[:, :obs_shape_real[0]] = obs
        
        
    rollouts.obs[0].copy_(new_obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    if task_idx > 0 and args.approach == 'gs':
        agent.freeze_init(task_idx)
    
    train_ppo(actor_critic, agent, rollouts, task_idx, env_name, task_sequences, envs, new_obs, obs_shape, obs_shape_real, args, 
              episode_rewards, tr_reward_arr, te_reward_arr, num_updates, log_name, device)
    
    if args.approach == 'fine-tuning':
        if args.single_task == True:
            envs.close()
            break
        else:
            envs.close()
        
    elif args.approach == 'ewc':
        agent.update_fisher(agent, envs, rollouts, task_idx, env_name, new_obs, obs_shape_real, args)
        envs.close()
        
    elif args.approach == 'ucl':
        
        if args.single_task == True:
            envs.close()
            break
        else:
            if task_idx == 0:
                agent.update_saved()
            agent.update_old_actor_critic()
            envs.close()
            
            

########################################################################################################################



