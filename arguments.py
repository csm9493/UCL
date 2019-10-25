import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=True,
                        choices=['mnist2', 
                                 'pmnist', 
                                 'split_pmnist', 
                                 'row_pmnist', 
                                 'mixture', 
                                 'omniglot',
                                 'split_mnist',
                                 'split_notmnist', 
                                 'split_row_pmnist', 
                                 'split_cifar10_100', 
                                 'split_cifar100',
                                 'split_cifar100_20',
                                 'split_CUB200', 
                                 'split_tiny_imagenet', 
                                 'split_mini_imagenet', 
                                 'split_cifar10'], 
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['random', 
                                 'sgd', 
                                 'sgd-frozen', 
                                 'sgd_with_log', 
                                 'sgd_L2_with_log', 
                                 'lwf','lwf_with_log', 
                                 'lfl',
                                 'ewc', 
                                 'si', 
                                 'rwalk', 
                                 'mas', 
                                 'ucl', 
                                 'ucl_ablation', 
                                 'baye_fisher',
                                 'baye_hat', 
                                 'imm-mean', 
                                 'progressive', 
                                 'pathnet',
                                 'imm-mode', 
                                 'sgd-restart', 
                                 'joint', 
                                 'hat', 
                                 'hat-test'], 
                        help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD', 
                                 'SGD_momentum_decay', 
                                 'Adam'], 
                        help='(default=%(default)s)')
    parser.add_argument('--ablation', default='None', type=str, required=False,
                        choices=['no_L1', 
                                 'no_upper', 
                                 'no_lower',
                                 'no_sigma_normal',
                                 'None'], 
                        help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--unitN', default=400, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lr_rho', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--ratio', default='0.5', type=float, help='(default=%(default)f)')
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default='0.03', type=float, help='(default=%(default)f)')
    parser.add_argument('--gamma', default=0.75, type=float, help='(default=%(default)f)')
    parser.add_argument('--smax', default=400, type=float, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--c', default='0.9', type=float, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=50, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--rebuttal', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')
    parser.add_argument('--rho', type = float, default=-2.783, help='initial rho')
    args=parser.parse_args()
    return args