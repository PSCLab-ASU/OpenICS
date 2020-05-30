import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
import utils


def reconstruction_method(reconstruction,specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if(reconstruction == 'LearnedDAMP' or reconstruction == 'LDAMP'):
        ldamp_wrap = LDAMP_wrapper()
        return ldamp_wrap

class LDAMP_wrapper():
    def initialize(self, dset,sensing_method,specifics):
        self.dset = dset
        self.sensing_method = sensing_method
        self.specifics = specifics
        self.LDAMPnet = LearnedDAMP(specifics)

    def run(self, stage):
        if (stage == 'training'):
            # pre-process training data
            x_true = self.dset
            y_measured = self.sensing_method(x_true) # returns noisy measurements (no noise if testing)
            # torch.save(LDAMP, './savedModels/ldamp1')
            return 1

        elif (stage == 'testing'):
            # pre-process training data
            x_test = self.dset
            y_measured = self.sensing_method(x_test) # returns noisy measurements (no noise if testing)
            self.LDAMPnet = torch.load('./savedModels/ldamp1') # load the model you want to test
            self.LDAMPnet.eval()
            x_hat = self.LDAMPnet(y_measured)

            return x_hat

class LearnedDAMP(nn.Module):
    def __init__(self, specifics):
        super(LearnedDAMP, self).__init__()
        # Define some useful functions
        def dncnnForward(r):
            channel_img = specifics["input_channel"]
            width_img = specifics["input_width"]
            height_img = specifics["input_height"]

            r = torch.t(r)
            orig_Shape = r.shape
            shape4D = [-1, channel_img, width_img, height_img]  # shape4D = [-1, height_img, width_img, channel_img]
            r = torch.reshape(r, shape4D)  # reshaping input
            x = F.relu(self.conv0(r))
            for i, l in enumerate(self.layers):
                x = l(x)
            x = F.relu(self.convLast(x))
            x_hat = r - x
            x_hat = torch.t(torch.reshape(x_hat, orig_Shape))
            return x_hat

        # instantiate functions and carry-over variables
        self.dncnnForward = dncnnForward
        self.m = specifics["m"]
        self.n = specifics["n"]
        self.BATCH_SIZE = specifics["BATCH_SIZE"]
        self.n_DAMP_layers = specifics["max_n_DAMP_layers"]
        self.A_val = utils.generateAVal(self.m,self.n)
        self.z = 0 # used in the LDAMP calculation between layers

        # instantiate denoisers
        n_DnCNN_layers = specifics["n_DnCNN_layers"]
        self.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        for i in range(1, n_DnCNN_layers - 1):
            conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            batchnorm = nn.BatchNorm2d(64)  # training
            self.layers.add_module("conv" + str(i), conv)
            self.layers.add_module("batch" + str(i), batchnorm)
        self.convLast = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, y_measured):
        self.z = y_measured # size = [m, BATCH_SIZE]
        xhat = torch.zeros([self.n, self.BATCH_SIZE], dtype=torch.float32)
        for iter in range(self.n_DAMP_layers):  # n_DAMP_layers should be 10
            r = xhat + utils.At_handle(self.A_val,self.z)
            x_hat = self.dncnnForward(r)

            r_abs = torch.abs(r)
            r_absColumnWise = torch.max(r_abs, dim=0)[0]
            epsilon = torch.max(r_absColumnWise * 0.001, torch.ones(r_absColumnWise.shape) * 0.00001)
            eta = torch.normal(0, 1, r.shape)  # Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample(r.shape)
            r_perturbed = r + torch.mul(eta, epsilon)
            xhat_perturbed = self.dncnnForward(r_perturbed)
            eta_dx = torch.mul(eta, xhat_perturbed - xhat)  # Want element-wise multiplication
            mean_eta_dx = torch.mean(eta_dx, dim=0)
            dxdr = torch.div(mean_eta_dx, epsilon)

            self.z = torch.Tensor(y_measured) - utils.A_handle(self.A_val, xhat)
            self.z = self.z + (float(self.n) / float(self.m) * dxdr * self.z)
        return x_hat
