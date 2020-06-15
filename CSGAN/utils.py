import torchvision
import torch
import numpy as np
import skimage.measure as skim

def generate_dataset(dataset,input_channel,input_width,input_height,stage):
    """Get the dataset as numpy arrays."""
    def preprocess(self, x):
        return x * 2 - 1
    # Construct the dataset.

    # mnist by default
    x = torch.load(dataset)
    # Note: tf dataset is binary so we convert it to float.
    x = x.type(torch.float32)
    x = x / 255.
    x = x.reshape((-1, 28, 28, 1))

    type = 'cifar'
    if type == 'cifar':
        x = torch.load(dataset)
        x = x.type(torch.float32)
        x = x / 255.

    # Normalize data if a processor is given.
    x = preprocess(x)
    return x

def addNoise(clean, sigma):
    clean = torch.tensor(clean)
    noise_vec = torch.rand(clean.shape)
    noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
    noisy = clean + noise_vec
    return noisy

def compute_average_psnr(img,img_hat):
    sz=img.size(0)
    return sum([skim.compare_psnr(img[i,:,:,:].numpy()/2.0+0.5,img_hat[i,:,:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz)])/sz