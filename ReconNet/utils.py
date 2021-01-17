import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import skimage.metrics as skimetrics
from PIL import Image

import os
import math
import string
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics):
    print("Creating dataset...")
    
    if stage == 'testing':
        root = specifics['test-root']
    elif stage == 'training':
        root = specifics['train-root']
    else:
        raise NotImplementedError

    dset = CustomDataset(
        root=root,
        transform=transforms.Compose(
            [
                transforms.Resize([input_height, input_width]),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * input_channel, [0.5] * input_channel)
            ]
        ),
        max_imgs=specifics['max-images'] if 'max-images' in specifics else None,
        name=dataset
    )
    
    print("Finished creating dataset")

    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    return dset

class CustomDataset(Dataset):
    def __init__(self, root, transform, name, max_imgs):
        self.root = root
        self.transform = transform
        self.all_imgs = os.listdir(root)[:max_imgs]
        self.name = name

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.all_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

class ReconNet(nn.Module):
    def __init__(self,indim,outdim1,outdim2,channel):
        super(ReconNet,self).__init__()
        self.indim=indim
        self.outdim1=outdim1
        self.outdim2=outdim2
        self.channel=channel
        self.init_layers=nn.ModuleList()
        
        for i in range(channel):
            self.init_layers.append(nn.Linear(indim, outdim1 * outdim2))
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
        xcs = []
        for i in range(self.channel):
            xcs.append(self.init_layers[i](x[:, i, :]))
        x=torch.cat(xcs,dim=1)
        x=x.view(-1,self.channel,self.outdim1,self.outdim2)
        x=F.relu(self.bn1(self.l1(x)))
        x=F.relu(self.bn2(self.l2(x)))
        x=F.relu(self.bn3(self.l3(x)))
        x=F.relu(self.bn4(self.l4(x)))
        x=F.relu(self.bn5(self.l5(x)))
        x=self.l6(x)
        return x

class random_sensing(nn.Module):
    def __init__(self,m,width,height,channel):
        super(random_sensing,self).__init__()
        self.m=m
        self.channel=channel
        self.width=width
        self.height=height
        self.s = nn.Linear(width * height, self.m, bias=False)
    def forward(self,x):
        if self.channel==3:
            m0 = self.s(x[:, 0, :, :].view(-1, self.width*self.height))
            m1 = self.s(x[:, 1, :, :].view(-1, self.width*self.height))
            m2 = self.s(x[:, 2, :, :].view(-1, self.width*self.height))
            m = torch.cat((m0, m1, m2), dim=1)
        else:
            m0 = self.s(x[:, 0, :, :].view(-1, self.width*self.height))
            m=m0
        x=m.view(-1,self.channel,self.m)
        return x

def compute_ssim(img,img_hat):
    return [skimetrics.structural_similarity(
        np.clip(img[i,:,:,:].numpy().T/2.0+0.5, 0.0, 1.0),
        np.clip(img_hat[i,:,:,:].numpy().T/2.0+0.5, 0.0, 1.0),
        multichannel=True,
        data_range=1.0
    ) for i in range(img.shape[0])]

def compute_psnr(img,img_hat):
    return [min(skimetrics.peak_signal_noise_ratio(
        np.clip(img[i,:,:,:].numpy().T/2.0+0.5, 0.0, 1.0),
        np.clip(img_hat[i,:,:,:].numpy().T/2.0+0.5, 0.0, 1.0),
        data_range=1.0
    ), 48.0) for i in range(img.shape[0])]

def random_name(N):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=N))

def create_dirs(model_root,logs_root,rname,stage,dataset,ratio):
    model_root = os.path.join(model_root, rname, dataset, 'cr' + str(ratio))
    logs_root = os.path.join(logs_root, rname, stage, dataset, 'cr' + str(ratio))
    
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    
    if not os.path.exists(logs_root):
        os.makedirs(logs_root)
    
    return model_root, logs_root

def save_imgs(img,img_hat,path):
    rows = math.floor(math.sqrt(len(img)))
    cols = rows * 2
    
    fig,axs = plt.subplots(rows, cols)
    
    for i in range(rows):
        for j in range(rows):
            ax = axs[i,rows + j]
            ax.set_axis_off()
            ax_hat = axs[i,j]
            ax_hat.set_axis_off()
            cur_img = img[i * rows + j]
            cur_img_hat = img_hat[i * rows + j]
            
            if cur_img.shape[0] == 1:
                cur_img = cur_img.squeeze()
                cur_img_hat = cur_img_hat.squeeze()
                ax.imshow(cur_img, cmap='gray')
                ax_hat.imshow(cur_img_hat, cmap='gray')
            else:
                ax.imshow(np.rot90(cur_img.T, -1))
                ax_hat.imshow(np.rot90(cur_img_hat.T, -1))
    
    fig.savefig(path)
