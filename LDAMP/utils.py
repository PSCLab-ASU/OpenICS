import torchvision
import torch
import numpy as np
import skimage.measure as skim

def generate_dataset(dataset,input_channel,input_width,input_height,stage):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(type(dataset) == str):
        images = np.load(dataset)
        images = images[:, 0, :, :]
        images = np.transpose(np.reshape(images, (-1, input_channel * input_height * input_width)))
    else:
        images = dataset
    return images


def A_handle(A_vals_tf, x):
    return torch.matmul(torch.Tensor(A_vals_tf), x.clone().detach())

def At_handle(A_vals_tf, z):
    return torch.matmul(torch.t(torch.conj(torch.Tensor(A_vals_tf))), z.clone().detach())

def generateAVal(m,n):
    return np.float32(1.0 / np.sqrt(float(m)) * np.random.randn(m, n))

def compute_average_psnr(img,img_hat):
    sz=img.size(0)
    return sum([skim.compare_psnr(img[i,:,:,:].numpy()/2.0+0.5,img_hat[i,:,:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz)])/sz