import sys
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla,unitN=400, split = False, notMNIST = False):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla=taskcla
        self.split = split
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,unitN)
        self.efc1=torch.nn.Embedding(len(self.taskcla),unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)
        self.efc2=torch.nn.Embedding(len(self.taskcla),unitN)
        if notMNIST:
            self.fc3=torch.nn.Linear(unitN,unitN)
            self.efc3=torch.nn.Embedding(len(self.taskcla),unitN)
            self.fc4=torch.nn.Linear(unitN,unitN)
            self.efc4=torch.nn.Embedding(len(self.taskcla),unitN)
        
        if split:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))
        else:
            self.fc3=torch.nn.Linear(unitN,taskcla[0][1])
        self.gate=torch.nn.Sigmoid()
        
        
        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        
        # Gated
        h=self.drop2(x.view(x.size(0),-1))
        h=self.drop1(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop1(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        if self.notMNIST:
            gfc3=self.gate(s*self.efc3(t))
            gfc4=self.gate(s*self.efc4(t))
            h=self.drop1(self.relu(self.fc3(h)))
            h=h*gfc3.expand_as(h)
            h=self.drop1(self.relu(self.fc4(h)))
            h=h*gfc4.expand_as(h)
        
        if self.split:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))
        else:
#             y=self.relu(self.fc3(h))
            y=self.fc3(h)
            
        masks = [gfc1, gfc2]
        if self.notMNIST:
            mask = [gfc1, gfc2, gfc3, gfc4]
        
        return y,masks

    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        if self.notMNIST:
            gfc3=self.gate(s*self.efc3(t))
            gfc4=self.gate(s*self.efc4(t))
            return [gfc1, gfc2, gfc3, gfc4]

        return [gfc1, gfc2]
        

    def get_view_for(self,n,masks):
        if self.notMNIST:
            gfc1,gfc2,gfc3,gfc4=masks
        else:
            gfc1,gfc2=masks
        
        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        if self.notMNIST:
            if n=='fc3.weight':
                post=gfc3.data.view(-1,1).expand_as(self.fc3.weight)
                pre=gfc2.data.view(1,-1).expand_as(self.fc3.weight)
                return torch.min(post,pre)
            elif n=='fc3.bias':
                return gfc3.data.view(-1)
            elif n=='fc4.weight':
                post=gfc4.data.view(-1,1).expand_as(self.fc4.weight)
                pre=gfc3.data.view(1,-1).expand_as(self.fc4.weight)
                return torch.min(post,pre)
            elif n=='fc4.bias':
                return gfc4.data.view(-1)

        
        return None

