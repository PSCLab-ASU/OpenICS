import torch
import scipy.io as sio
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


def generate_dataset(dataset, type, input_channel, input_width, input_height, stage):
    # a function to generate the corresponding dataset with given parameters.
    # return an instance of the dataset class.

    # number of training blocks
    nrtrain = 88912

    if type == "mat":
        data = sio.loadmat(dataset)
        labels = data["labels"]
        return RandomDataset(labels, nrtrain)
