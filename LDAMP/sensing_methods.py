import torch
from torch import nn
import torchvision
import numpy as np


def sensing_method(sensing,specifics,m,n):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    img_dimension = specifics["input_width"]
    channel = specifics["input_channel"]
    from_Numpy_matrix = np.float32(1.0 / np.sqrt(m) * np.random.randn(m, n))
    from_Numpy_matrix = torch.Tensor(from_Numpy_matrix)
    sense = random_sensing(n, m, img_dimension, channel, from_Numpy_matrix)
    torch.conj(from_Numpy_matrix)
    return sense


class random_sensing(nn.Module):
    def __init__(self, n, m, imgdim, channel, from_Numpy_matrix):
        super(random_sensing, self).__init__()
        self.n = n
        self.m = m
        self.channel = channel
        self.imgdim = imgdim
        sensing_matrix = from_Numpy_matrix[:m, :n]

        self.sm = torch.from_numpy(sensing_matrix).cuda()
        # if (channel == 1):
        self.s1 = nn.Linear(n, m, bias=False)
        self.s1.weight.data = self.sm[0:m, :n]
        # elif (channel == 3):
        #     self.s1 = nn.Linear(imgdim * imgdim, m, bias=False)
        #     self.s1.weight.data = self.sm[0, 0:m, :]
        #     self.s2 = nn.Linear(imgdim * imgdim, m, bias=False)
        #     self.s2.weight.data = self.sm[1, 0:m, :]
        #     self.s3 = nn.Linear(imgdim * imgdim, m, bias=False)
        #     self.s3.weight.data = self.sm[2, 0: m, :]


    def forward(self, x):
        # if self.channel == 1:
        r = self.s1(x)
        # else:
        #     r1 = self.s1(x[:, 0, :])
        #     r2 = self.s1(x[:, 1, :])
        #     r3 = self.s1(x[:, 2, :])
        #     r = torch.cat([r1, r2, r3], dim=1)
        '''
        if channel == 0:
            r = self.s1(x)
        elif channel ==1:
            r = self.s2(x)
        elif channel ==2:
            r = self.s3(x)
        '''
        return r

    def returnSensingMatrix(self):
        return self.sm