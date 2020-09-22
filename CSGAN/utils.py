import torchvision
from torchvision import datasets, transforms
import numpy as np
import skimage.measure as skim
import torch

def generate_dataset(dataset,input_channel,input_width,input_height,stage):
    trans = PreprocessTransform()
    if stage == "testing":
        test_dataset = None
        if dataset == 'mnist':
            test_dataset = datasets.MNIST('./data', train=False, download = True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                        ]))
        elif dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                            ]))
        elif dataset == 'celeba':
            test_dataset = datasets.CelebA(root='./data', split="test", download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                            ]))
        return test_dataset
    elif stage == "training":
        train_dataset = None
        val_dataset = None
        if dataset == 'mnist':
            train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Lambda(trans)                                            
                                        ]))
            val_dataset = datasets.MNIST(root='./data/val', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                        ]))
        elif dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='./data',train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                                transforms.Lambda(trans)
                                            ]))

            val_dataset = datasets.CIFAR10(root='./data',train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                        ]))
        elif dataset == 'celeba':
            train_dataset = datasets.CelebA(root='./data',split="train", download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor(),
                                                transforms.Lambda(trans)
                                            ]))

            val_dataset = datasets.CelebA(root='./data',split="valid", download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_width),
                                            transforms.CenterCrop(input_width),
                                            transforms.ToTensor(),
                                            transforms.Lambda(trans)
                                        ]))
        return (train_dataset, val_dataset)

class PreprocessTransform: 
    def __call__(self, x):
        x = x*2-1
        return x
def postprocess(x):
    x = (x+1)/2
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

def make_prior(num_latents):
    prior_mean = torch.zeros(num_latents, dtype=torch.float32)
    prior_scale = torch.ones(num_latents, dtype=torch.float32)

    return torch.distributions.normal.Normal(prior_mean,prior_scale)

