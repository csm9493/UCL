import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0,pc_valid=0.10, tasknum = 10):
    data={}
    taskcla=[]
    size=[3,32,32]

    if not os.path.isdir('../dat/binary_split_cifar100/'):
        os.makedirs('../dat/binary_split_cifar100')
        os.makedirs('../dat/binary_cifar10')

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
        
        # CIFAR10
        dat={}
        dat['train']=datasets.CIFAR10('../dat/',train=True,download=True,
                                      transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('../dat/',train=False,download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        data[0]={}
        data[0]['name']='cifar10'
        data[0]['ncla']=10
        data[0]['train']={'x': [],'y': []}
        data[0]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(target.numpy()[0])
        
        
        # CIFAR100
        dat={}
        
        dat['train']=datasets.CIFAR100('../dat/',train=True,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('../dat/',train=False,download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for n in range(1,11):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                task_idx = target.numpy()[0] // 10 + 1
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0]%10)

        # "Unify" and save
        for s in ['train','test']:
            data[0][s]['x']=torch.stack(data[0][s]['x']).view(-1,size[0],size[1],size[2])
            data[0][s]['y']=torch.LongTensor(np.array(data[0][s]['y'],dtype=int)).view(-1)
            torch.save(data[0][s]['x'], os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'x.bin'))
            torch.save(data[0][s]['y'], os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'y.bin'))
        for t in range(1,11):
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                         'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                         'data'+str(t)+s+'y.bin'))
    
    # Load binary files
    data={}
    data[0] = dict.fromkeys(['name','ncla','train','test'])
    for s in ['train','test']:
        data[0][s]={'x':[],'y':[]}
        data[0][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'x.bin'))
        data[0][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar10'),'data'+s+'y.bin'))
    data[0]['ncla']=len(np.unique(data[0]['train']['y'].numpy()))
    data[0]['name']='cifar10'
    
    ids=list(shuffle(np.arange(10),random_state=seed) + 1)
#     ids=list(range(1,11))
    print('Task order =',ids)
    for i in range(1,11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                    'data'+str(ids[i-1])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'),
                                                    'data'+str(ids[i-1])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name']='cifar100-'+str(ids[i-1])
            
    # Validation
    for t in range(11):
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in range(11):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size