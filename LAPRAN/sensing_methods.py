import torch
import torchvision
import torch.nn as nn
import numpy as np
def sensing_method(method_name,n,m,imgdim,channel,path = None):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    sensing_method = None
    if method_name == "random_sensing":
        sensing_method = random_sensing
    return sensing_method


class random_sensing(nn.Module):
    def __init__(self,n, m,imgdim,channel, from_Numpy_matrix ):
        super(random_sensing,self).__init__()
        self.n = n
        self.m = m
        self.channel=channel
        self.imgdim=imgdim
        
        sensing_matrix = from_Numpy_matrix[:,:m,:]

        sm=torch.from_numpy(sensing_matrix).float().cuda()
        if channel==1:
            self.s1 = nn.Linear(imgdim * imgdim, m, bias=False, )
            self.s1.weight.data = sm[0, 0:m, :]
        elif channel==3:
            self.s1 = nn.Linear(imgdim * imgdim,  m, bias=False, )
            self.s1.weight.data = sm[0, 0:m, :]
            self.s2 = nn.Linear(imgdim * imgdim,m, bias=False)
            self.s2.weight.data = sm[1, 0:m, :]
            self.s3 = nn.Linear(imgdim * imgdim, m, bias=False)
            self.s3.weight.data = sm[2, 0: m, :]
        
    def forward(self,x):
        if self.channel==1:
            r = self.s1(x)
        else:
            r1=self.s1(x[:,0,:])
            r2=self.s2(x[:,1,:])
            r3=self.s3(x[:,2,:])
            r=torch.cat([r1,r2,r3],dim=1)
        '''
        if channel == 0:
            r = self.s1(x)
        elif channel ==1:
            r = self.s2(x)
        elif channel ==2:
            r = self.s3(x)
        '''
        return r