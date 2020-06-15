import torch
import torchvision
import numpy as np


def sensing_method(sensing,specifics,m,n):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    sigma_w = specifics["sigma_w"]
    if(not (sensing == "Gaussian")):
        print('Sensing method "' + sensing + '" is unsupported for LDAMP. Default Gaussian will be used')

    A_val = np.float32(1.0 / np.sqrt(m) * np.random.randn(m, n))
    def sense(x):
        y = torch.matmul(torch.Tensor(A_val), x)
        noisy_data = addNoise(y, sigma_w)
        return noisy_data

    def addNoise(clean, sigma):
        clean = torch.tensor(clean)
        noise_vec = torch.rand(clean.shape)
        noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
        noisy = clean + noise_vec
        return noisy

    return sense