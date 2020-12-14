import torch
import torchvision
import torch.nn as nn
import numpy as np
def sensing_method(method_name,n,m,imgdim,channel,path = None):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    sensing_method = None
    if method_name == "sensing_matrix":
        sensing_method = random_sensing
    elif method_name == "NN_MLP":
        sensing_method = MLPMetricNet
    elif method_name == "NN_DCGAN":
        sensing_method = csgm_dcgan_disc
    return sensing_method


class random_sensing(nn.Module):
    def __init__(self,n, m,imgdim,channel ):
        super(random_sensing,self).__init__()
        self.n = n
        self.m = m
        self.channel=channel
        self.imgdim=imgdim
        
        sm = torch.normal(mean = 0.0, std = 0.05, size = [self.m,self.n]).cuda()

        self.s1 = nn.Linear(n, m, bias=False)
        self.s1.weight.data = sm[0:m, :]

        
    def forward(self,x):
        x = x.reshape(-1,self.n)
        r = self.s1(x)
        output = torch.reshape(r,[-1,self.m])
        return output
class MLPMetricNet(nn.Module):
  def __init__(self, n, m,imgdim,channel, name='mlp_metric'):
    super(MLPMetricNet, self).__init__() #super(MLPMetricNet, self).__init__(name=name)
    #self.linear = nn.Linear(500, 500, num_outputs, bias=True)
    self.main = nn.Sequential(
        nn.Linear(n,500,True),
        nn.LeakyReLU(0.2, inplace=False),
        nn.Linear(500,500,True),
        nn.LeakyReLU(0.2,inplace=False),
        nn.Linear(500,m,True)
    )
  def forward(self, inputs):
    inputs = torch.flatten(inputs,start_dim=1)
    output = self.main(inputs)
    return output
class SNMetricNet(nn.Module):
  """CelebA discriminator"""
  def __init__(self, n, m,imgdim,channel, name='snmetricnet'):
    super(SNMetricNet, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.main = nn.Sequential(
        nn.Conv2d(3,64,(3,3),1,padding=1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(64,64,(4,4),2, padding = 1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(64,128,(3,3),1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(128,128,(4,4),2, padding = 1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(128,256,(3,3),1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(256,256,(4,4),2, padding = 1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(256,512,(3,3),1, padding = 1),
        nn.LeakyReLU(0.1) #the output at this point is (batchsize, 512,8,8) (for imgdim 64, because size is halved three times)
    )
    self.lin = nn.Linear(32768,m,True)

  def forward(self, inputs):
 
    outputs = self.main(inputs)
 
    outputs = self.lin(torch.flatten(outputs,start_dim=1))
    return outputs
class csgm_dcgan_disc(nn.Module):
  """CelebA discriminator"""
  def __init__(self, n, m,imgdim,channel, name='snmetricnet2'):
    super(csgm_dcgan_disc, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.main = nn.Sequential(
        nn.Conv2d(3,64,(5,5),2,padding=1),
        nn.ZeroPad2d((0,1,0,1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64,128,(5,5),2, padding = 1),
        nn.ZeroPad2d((0,1,0,1)),
        nn.BatchNorm2d(128,1e-5,0.9,True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128,256,(5,5),2, padding = 1),
        nn.ZeroPad2d((0,1,0,1)),
        nn.BatchNorm2d(256,1e-5,0.9,True),
        nn.LeakyReLU(0.2,inplace =True),
        nn.Conv2d(256,512,(5,5),2, padding = 1),
        nn.ZeroPad2d((0,1,0,1)),
        nn.BatchNorm2d(512,1e-5,0.9,True),
        nn.LeakyReLU(0.2,inplace=True) #the output at this point is (batchsize, 512,4,4 (for imgdim 64, because size is halved three times)
    )
    self.lin = nn.Sequential(
        nn.Linear(8192,m,True)
        #nn.Sigmoid()
        )
  def forward(self, inputs):
 
    outputs = self.main(inputs)
    outputs = self.lin(torch.flatten(outputs,start_dim=1))
    return outputs
