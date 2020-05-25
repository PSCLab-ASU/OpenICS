import torchvision
import torch
import numpy as np

def generate_dataset(dataset,input_channel,input_width,input_height,stage):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    images = np.load(dataset)
    images = images[:, 0, :, :]
    images = np.transpose(np.reshape(images, (-1, input_channel * input_height * input_width)))
    return images


def A_handle(A_vals_tf, x):
    return torch.matmul(torch.Tensor(A_vals_tf), x.clone().detach())

def At_handle(A_vals_tf, z):
    return torch.matmul(torch.t(torch.conj(torch.Tensor(A_vals_tf))), z.clone().detach())

def generateAVal(m,n):
    return np.float32(1.0 / np.sqrt(float(m)) * np.random.randn(m, n))
