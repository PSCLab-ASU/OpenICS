import torch
import torchvision
import utils as u
def sensing_method(method_name,specifics):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    return u.random_sensing(specifics['m'],specifics['input_width'],specifics['input_channel'])
