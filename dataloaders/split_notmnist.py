import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor

"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.
Set root to point to the Train/Test folders.
"""
# (A,F), (B,G), (C,H), (D,I), (E,J)
def split_notMNIST_loader(root):
    
    data = {}
    for i in range(5):
        data[i] = {}
        data[i]['name'] = 'split_notmnist-{:d}'.format(i)
        data[i]['ncla'] = 2
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}

    folders = os.listdir(root)
    task_cnt = 0
    for folder in folders:
        folder_path = os.path.join(root, folder)
        cnt=0
        print(folder)
        for ims in os.listdir(folder_path):
            s = 'train'
            if cnt >= 40000:
                s = 'test'
            try:
                img_path = os.path.join(folder_path, ims)
                img = imread(img_path) / 255.0
                img_tensor = Tensor(img).float()
                task_idx = (ord(folder) - 65) % 5
                label = (ord(folder) - 65) // 5
                data[task_idx][s]['x'].append(img_tensor)
                data[task_idx][s]['y'].append(label) # Folders are A-J so labels will be 0-9
                cnt += 1

            except:
                # Some images in the dataset are damaged
                print("File {}/{} is broken".format(folder, ims))
        task_cnt += 1
    
    return data


def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 5):
    if tasknum>5:
        tasknum = 5
    
    data = {}
    taskcla = []
    size = [1, 28, 28]
    
    
    # Pre-load
    # notMNIST
#     mean = (0.1307,)
#     std = (0.3081,)
    if not os.path.isdir('../dat/binary_split_notmnist/'):
        os.makedirs('../dat/binary_split_notmnist')
        root = os.path.dirname(__file__)
        data = split_notMNIST_loader(os.path.join(root, '../../dat/notMNIST_large'))
        
        for i in range(5):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x'])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_split_notmnist'),
                                                        'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_split_notmnist'),
                                                        'data' + str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(5):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 2
            data[i]['name'] = 'split_notmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_notmnist'),
                                                          'data' + str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_notmnist'),
                                                          'data' + str(i) + s + 'y.bin'))
        
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, size

