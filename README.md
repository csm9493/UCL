

# [Uncertainty-based Continual Learning with Adaptive Regularization (UCL)](https://papers.nips.cc/paper/8690-uncertainty-based-continual-learning-with-adaptive-regularization), published at NeurIPS 2019

**Hongjoon Ahn, Sungmin Cha, Donggyu Lee, and Taesup Moon**

**[M.IN.D Lab](https://mindlab-skku.github.io), Sungkyunkwan University**

------

## Running UCL on various dataset

#### To run UCL on pmnist, enter the following command:

```
$ python3 main.py --experiment pmnist --approach ucl --beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01 
```

#### To run UCL on split-CIFAR10/100, enter the following command:

```
$ python3 main.py --experiment split_cifar10_100 --approach ucl --conv-net --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3
```

#### To run UCL on split-CIFAR100, enter the following command:

```
$ python3 main.py --experiment split_cifar100 --approach ucl --conv-net --beta 0.002 --ratio 0.125 --lr_rho 0.01 --alpha 5 
```

#### To run UCL on Omniglot, enter the following command:

```
$ python3 main.py --experiment omniglot --approach ucl --conv-net --beta 0.00001 --ratio 0.5 --lr_rho 0.02 --alpha 5
```

#### Note that, in all commands. alpha is the weight decay penalty strength for first task. And, all the commands are based on our new initialization technique, which is introduced in Appendix of our paper.

#### This repository also contains the implementations for baselines, such as EWC, SI, RWALK, MAS, and HAT.

#### All the results of experiments are saved at ./result_data in txt file format.

------

## Running UCL on Roboschool

### Requirements

- Python 3.6
- Pytorch 1.2.0+cu9.2 / CUDA 9.2
- OpenAI Gym, Baselines, Roboschool

#### Notes

This code is implemented by reference to [pytorch-a2c-ppo-acktr-gaail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 

#### 1) Install OpenAI Gym, Baselines, Roboschool

​	Follow below links for installation

​	[OpenAI Gym](https://github.com/openai/gym#installation), [Baselines](https://github.com/openai/baselinesn), [Roboschool](https://github.com/openai/roboschool)

#### 3) Execution command

```
# Fine-tuning
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --experiment 'roboschool'  --approach ‘fine-tuning’  --date 191014

# EWC
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --experiment 'roboschool'  --approach 'ewc'  --ewc-lambda 5000 --date 191014

# AGS-CL
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --experiment 'roboschool'  --approach ‘ucl’  --ucl-rho -2.2522 -ucl-beta 0.001 --date 191014
```

## **Citation**

```
@inproceedings{ahn2019uncertainty,
  title={Uncertainty-based continual learning with adaptive regularization},
  author={Ahn, Hongjoon and Cha, Sungmin and Lee, Donggyu and Moon, Taesup},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4394--4404},
  year={2019}
}
```

Reference

1. Bayesian neural network implementation has been modified from: https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
2. The whole experiment framework has been modified from: https://github.com/joansj/hat
