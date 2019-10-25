import sys
import torch
import torch.nn as nn
from utils import *

class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize #28
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3)
        s = compute_conv_output_size(size,3) #26
        self.conv2 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #24
        s = s//2 #12
        self.conv3 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #10
        self.conv4 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #8
        s = s//2 #4
        
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(s*s*64,n)) #4*4*64 = 1024
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.MaxPool(h)
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.MaxPool(h)
        h=h.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y