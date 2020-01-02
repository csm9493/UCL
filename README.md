
UCL(Published at NeurIPS 2019)
================================

### This repository is implementation for the paper [Uncertainty-based Continual Learning with Adaptive Regularization](https://papers.nips.cc/paper/8690-uncertainty-based-continual-learning-with-adaptive-regularization) by Hongjoon Ahn\*, Sungmin Cha\*, Donggyu Lee, and Taesup Moon

Running UCL on various dataset
--------------------------------


#### To run UCL on pmnist, enter the following command:


``` python3 main.py --experiment pmnist --approach ucl --beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01 ```

#### To run UCL on split-CIFAR10/100, enter the following command:

``` python3 main.py --experiment split_cifar10_100 --approach ucl --conv-net --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3 ```



#### To run UCL on split-CIFAR100, enter the following command:

``` python3 main.py --experiment split_cifar100 --approach ucl --conv-net --beta 0.002 --ratio 0.125 --lr_rho 0.01 --alpha 5  ```

#### To run UCL on Omniglot, enter the following command:

``` python3 main.py --experiment omniglot --approach ucl --conv-net --beta 0.00001 --ratio 0.5 --lr_rho 0.02 --alpha 5  ```

#### Note that, in all commands. alpha is the weight decay penalty strength for first task. And, all the commands are based on our new initialization technique, which is introduced in Appendix of our paper.

#### This repository also contains the implementations for baselines, such as EWC, SI, RWALK, MAS, and HAT.

#### All the results of experiments are saved at ./result_data in txt file format.

Note
=====
#### In this repository, it only contains the implementations for supervised learning experiments in UCL. We'll release the implementations for reinforcement learning experiments soon.

Reference
=========
1. Bayesian neural network implementation has been modified from: https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
2. The whole experiment framework has been modified from: https://github.com/joansj/hat
