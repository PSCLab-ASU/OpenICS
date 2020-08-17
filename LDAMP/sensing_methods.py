import torch
from torch import nn
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def sensing_method(sensing,specifics,m,n):
    sense = random_sensing(specifics)
    return sense

class random_sensing(nn.Module):
    def __init__(self, specifics):
        super(random_sensing, self).__init__()
        self.n = specifics["n"]
        self.m = specifics["m"]
        self.sigma = specifics["sigma_w"]
        self.s1 = nn.Linear(self.n, self.m, bias=False)
        self.from_Numpy_matrix = 0
        self.sm = 0

    def generateNewMatrix(self):
        self.from_Numpy_matrix = np.float32(1.0 / np.sqrt(self.m) * np.random.randn(self.m, self.n))
        self.sm = torch.from_numpy(self.from_Numpy_matrix).to(device)
        self.s1.weight.data = self.sm

    def forward(self, x):
        r = self.s1(x)
        noise = self.sigma * torch.rand(r.shape).to(device)
        r = r + noise
        return r

    def returnSensingMatrix(self):
        return self.sm

class original_sensing():
    def __init__(self, specifics, from_Numpy_matrix):
        self.sigma_w = specifics["sigma_w"]
        self.A_val = from_Numpy_matrix

    def sense(self, x):
        y = torch.matmul(torch.Tensor(self.A_val).to(device), torch.t(x.type(torch.FloatTensor).to(device)))
        clean = torch.tensor(y)
        noise_vec = torch.rand(clean.shape)
        noise_vec = self.sigma_w * np.reshape(noise_vec, newshape=clean.shape)
        noisy_data = clean + noise_vec.to(device)
        return noisy_data

    def returnSensingMatrix(self):
        return torch.Tensor(self.A_val)