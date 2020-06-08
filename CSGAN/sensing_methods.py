import torch
import torchvision
from torch import nn


def sensing_method(method_name,specifics, m, n):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    if(not (method_name == "Gaussian")):
        print('Sensing method "' + method_name + '" is unsupported for LDAMP. Default Gaussian will be used')

    # Measurement matrix
    A = torch.randn(m, n)

    # Gaussian sensing
    def sense(x):
        y = torch.matmul(A, x)
        return y

    return sense
