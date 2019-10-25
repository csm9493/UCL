import sys
import torch
import torch.nn as nn
from utils import *

class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize #28
        self.taskcla = taskcla
        
        self.c1 = nn.Conv2d(ncha,64,kernel_size=3)
        s = compute_conv_output_size(size,3) #26
        self.c2 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #24
        s = s//2 #12
        self.c3 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #10
        self.c4 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #8
        s = s//2 #4
        
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(s*s*64,n)) #4*4*64 = 1024
        self.relu = torch.nn.ReLU()
        
        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),64)
        self.ec3=torch.nn.Embedding(len(self.taskcla),64)
        self.ec4=torch.nn.Embedding(len(self.taskcla),64)
        
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.ec4.weight.data.uniform_(lo,hi)
        #"""

    def forward(self, x):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4=masks
        
        #Gated
        h=self.relu(self.c1(x))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c2(h))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.MaxPool(h)
        h=self.relu(self.c3(x))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c4(h))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=self.MaxPool(h)
        
        h=h.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y
    
    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        return [gc1,gc2,gc3,gc4]
    
    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gc4=masks
        
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        elif n=='c4.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.c4.weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.c4.weight)
            return torch.min(post,pre)
        elif n=='c4.bias':
            return gc4.data.view(-1)
        
        return None
