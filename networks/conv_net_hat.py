import sys
import torch
import torch.nn as nn
from utils import *

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.c1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.c2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.c3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.c4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.c5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.c6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
#         self.c7 = nn.Conv2d(128,128,kernel_size=3,padding=1)
#         s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        
        self.smid=s
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),32)
        self.ec2=torch.nn.Embedding(len(self.taskcla),32)
        self.ec3=torch.nn.Embedding(len(self.taskcla),64)
        self.ec4=torch.nn.Embedding(len(self.taskcla),64)
        self.ec5=torch.nn.Embedding(len(self.taskcla),128)
        self.ec6=torch.nn.Embedding(len(self.taskcla),128)
#         self.ec7=torch.nn.Embedding(len(self.taskcla),128)
        self.efc1=torch.nn.Embedding(len(self.taskcla),256)
        
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.ec4.weight.data.uniform_(lo,hi)
        self.ec5.weight.data.uniform_(lo,hi)
        self.ec6.weight.data.uniform_(lo,hi)
        self.ec7.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
#         gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1=masks
        gc1,gc2,gc3,gc4,gc5,gc6,gfc1=masks
        
        # Gated
        h=self.relu(self.c1(x))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c2(h))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c3(h))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c4(h))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.c5(h))
        h=h*gc5.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.c6(h))
        h=h*gc6.view(1,-1,1,1).expand_as(h)
#         h=self.relu(self.c7(h))
#         h=h*gc7.view(1,-1,1,1).expand_as(h)
        h=self.drop1(self.MaxPool(h))
        
        h=h.view(x.shape[0],-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        gc5=self.gate(s*self.ec5(t))
        gc6=self.gate(s*self.ec6(t))
#         gc7=self.gate(s*self.ec7(t))
        gfc1=self.gate(s*self.efc1(t))
#         return [gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1]
        return [gc1,gc2,gc3,gc4,gc5,gc6,gfc1]

    def get_view_for(self,n,masks):
#         gc1,gc2,gc3,gc4,gc5,gc6,gc7,gfc1=masks
        gc1,gc2,gc3,gc4,gc5,gc6,gfc1=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc6.data.view(-1,1,1).expand((self.ec6.weight.size(1),
                                              self.smid,
                                              self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
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
        elif n=='c5.weight':
            post=gc5.data.view(-1,1,1,1).expand_as(self.c5.weight)
            pre=gc4.data.view(1,-1,1,1).expand_as(self.c5.weight)
            return torch.min(post,pre)
        elif n=='c5.bias':
            return gc5.data.view(-1)
        elif n=='c6.weight':
            post=gc6.data.view(-1,1,1,1).expand_as(self.c6.weight)
            pre=gc5.data.view(1,-1,1,1).expand_as(self.c6.weight)
            return torch.min(post,pre)
        elif n=='c6.bias':
            return gc6.data.view(-1)
#         elif n=='c7.weight':
#             post=gc7.data.view(-1,1,1,1).expand_as(self.c7.weight)
#             pre=gc6.data.view(1,-1,1,1).expand_as(self.c7.weight)
#             return torch.min(post,pre)
#         elif n=='c7.bias':
#             return gc7.data.view(-1)
        return None

