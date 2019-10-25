import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio):
        super().__init__()
        
        ncha,size,_=inputsize #28
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,64,kernel_size=3,ratio=ratio)
        s = compute_conv_output_size(size,3) #26
        self.conv2 = BayesianConv2D(64,64,kernel_size=3,ratio=ratio)
        s = compute_conv_output_size(s,3) #24
        s = s//2 #12
        self.conv3 = BayesianConv2D(64,64,kernel_size=3,ratio=ratio)
        s = compute_conv_output_size(s,3) #10
        self.conv4 = BayesianConv2D(64,64,kernel_size=3,ratio=ratio)
        s = compute_conv_output_size(s,3) #8
        s = s//2 #4
        
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(s*s*64,n)) #4*4*64 = 1024
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False):
        h=self.relu(self.conv1(x,sample))
        h=self.relu(self.conv2(h,sample))
        h=self.MaxPool(h)
        h=self.relu(self.conv3(h,sample))
        h=self.relu(self.conv4(h,sample))
        h=self.MaxPool(h)
        h=h.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y