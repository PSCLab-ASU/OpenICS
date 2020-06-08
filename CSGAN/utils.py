import torchvision
import torch
import numpy as np

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

