import torchvision
import torch.nn as nn
import torch.functional as F
import torch
import skimage.measure as skim
def generate_dataset():
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    return 1

class reconnet(nn.Module):
    def __init__(self,indim,outdim,channel):
        super(reconnet,self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.channel=channel
        if channel==3:
            self.initlayer0 = nn.Linear(indim, outdim * outdim)
            self.initlayer1 = nn.Linear(indim, outdim * outdim)
            self.initlayer2 = nn.Linear(indim, outdim * outdim)
        else:
            self.initlayer0 = nn.Linear(indim, outdim * outdim)
        self.l1=nn.Conv2d(self.channel,64,11,padding=5)
        self.bn1=nn.BatchNorm2d(64)
        self.l2=nn.Conv2d(64,32,1)
        self.bn2=nn.BatchNorm2d(32)
        self.l3=nn.Conv2d(32,self.channel,7,padding=3)
        self.bn3=nn.BatchNorm2d(self.channel)
        self.l4 = nn.Conv2d(self.channel, 64, 11, padding=5)
        self.bn4=nn.BatchNorm2d(64)
        self.l5 = nn.Conv2d(64, 32, 1)
        self.bn5=nn.BatchNorm2d(32)
        self.l6 = nn.Conv2d(32, self.channel, 7, padding=3)
    def forward(self,x):
        sz=x.size()
        if self.channel==1:
            xc0 = self.initlayer0(x[:, 0, :])
            x=xc0
        else:
            xc0 = self.initlayer0(x[:, 0, :])
            xc1 = self.initlayer1(x[:, 1, :])
            xc2 = self.initlayer2(x[:, 2, :])
            x=torch.cat((xc0,xc1,xc2),dim=1)
        x=x.view(sz[0],self.channel,self.outdim,self.outdim)
        x=F.relu(self.bn1(self.l1(x)))
        x=F.relu(self.bn2(self.l2(x)))
        x=F.relu(self.bn3(self.l3(x)))
        x=F.relu(self.bn4(self.l4(x)))
        x=F.relu(self.bn5(self.l5(x)))
        x=self.l6(x)
        return x

class random_sensing(nn.Module):
    def __init__(self,nm,imgdim,channel):
        super(random_sensing,self).__init__()
        self.nm=nm
        self.channel=channel
        self.imgdim=imgdim
        sm=torch.randn(nm,imgdim*imgdim)/nm
        self.s = nn.Linear(imgdim * imgdim, self.nm, bias=False)
    def forward(self,x):
        bsz=x.size(0)
        if self.channel==3:
            m0 = self.s(x[:, 0, :, :].view(bsz, self.imgdim*self.imgdim))
            m1 = self.s(x[:, 1, :, :].view(bsz, self.imgdim*self.imgdim))
            m2 = self.s(x[:, 2, :, :].view(bsz, self.imgdim*self.imgdim))
            m = torch.cat((m0, m1, m2), dim=1)
        else:
            m0 = self.s(x[:, 0, :, :].view(bsz, self.imgdim*self.imgdim))
            m=m0
        x=m.view(bsz,self.channel,self.nm)
        return x

def compute_average_psnr(img,img_hat):
    sz=img.size(0)
    return sum([skim.compare_psnr(img[i,:,:,:].numpy()/2.0+0.5,img_hat[i,:,:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz)])/sz

