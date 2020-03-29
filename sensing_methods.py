import torch
# import torchvision
from torch.utils.data import DataLoader


def sensing_method(method_name, data, m, specifics):
    # a function which returns a sensing method with given parameters.
    # a sensing method is a subclass of nn.Module --
    #   this is not how istanet manages this, ill do it their way and then convert
    if method_name == "Gaussian":
        return DataLoader(
            dataset=data, batch_size=specifics["batch_size"], shuffle=True
        )
    return 1
