import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

def reconstruction_method(reconstruction,specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if(reconstruction == 'LearnedDAMP' or reconstruction == 'LDAMP'):
        ldamp_wrap = LDAMP_wrapper(specifics)
        return ldamp_wrap

class LDAMP_wrapper():
    def __init__(self, specifics):
        self.name = 'LearnedDAMPwrap'
        self.specifics = specifics

    def initialize(self, dset,sensing_method,specifics):
        self.BATCH_SIZE = specifics["BATCH_SIZE"]
        self.input_channel = specifics["input_channel"]
        self.input_width = specifics["input_width"]
        self.input_height = specifics["input_height"]
        self.n = specifics["n"]
        self.m = specifics["m"]

        self.dset = dset
        self.sensing_method = sensing_method
        self.net = LearnedDAMP(specifics=specifics)
        # print(self.net)
        self.n_train_images = self.specifics['n_train_images']
        self.n_val_images = self.specifics["n_val_images"]
        self.dataloader = torch.utils.data.DataLoader(self.dset[self.n_val_images:self.n_train_images],
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)
        self.testdataloader=torch.utils.data.DataLoader(self.dset[:self.n_val_images],
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)
        self.loss_f = torch.nn.MSELoss(reduction='mean')  # use 'mean' for torch versions after 1.0.0

        params = list(self.net.parameters())
        print("Number of parameters: " + str(len(params)))

    def run(self, stage):
        if (stage == 'training'):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.specifics['learning_rate'])
            history = []
            for epoch in range(self.specifics["EPOCHS"]):
                print("Beginning EPOCH " + str(epoch) + "...")
                for i, img in enumerate(self.dataloader):
                    print("Processing batch " + str(i) + "...")

                    # zero gradients
                    self.optimizer.zero_grad()

                    # set up inputs
                    A_val = self.sensing_method.returnSensingMatrix()
                    A_val = A_val.detach()
                    shape = (self.input_channel * self.input_width * self.input_height, self.BATCH_SIZE)
                    img = np.reshape(img, shape)
                    measurementMaybe = torch.matmul(A_val, img.type(torch.FloatTensor).cuda())

                    shape = (self.BATCH_SIZE, self.input_channel * self.input_width * self.input_height)
                    img = np.reshape(img, shape)
                    img = img.type(torch.float).cuda()

                    measurement = self.sensing_method(img)
                    shape = (self.m, self.BATCH_SIZE)#(self.input_channel, self.m, self.BATCH_SIZE)
                    measurement = np.reshape(measurement.cpu().detach().numpy(), shape)
                    measurement = torch.Tensor(measurement)

                    measurement = measurementMaybe
                    print(measurement)

                    # forward function
                    img_hat = self.net(measurement, A_val) # (y, z, A matrix) the z is overwritten so doesn't matter what it is
                    img_hat = torch.reshape(img_hat, img.shape)

                    # calculate loss and backpropogate
                    img = img.type(torch.FloatTensor).cpu()
                    loss = self.loss_f(img_hat, img) #torch.t(img.type(torch.FloatTensor).cpu()))
                    loss.backward()
                    self.optimizer.step()

                    # display some errors
                    _, _, error = utils.EvalError(img_hat, img)
                    history.append(error)
                    print('error (previous is left, most recent is right): ')
                    if (len(history) > 1):
                        print(history[len(history) - 2], history[len(history) - 1])
                    else:
                        print(history)
                    print

                    if (i % 50 == 0):
                        print("")
                        print("printing graphs...")

                        io = img.cpu().numpy()
                        shape = (self.BATCH_SIZE, self.input_width, self.input_height, self.input_channel)
                        sample = np.reshape(io, shape)
                        print(sample.shape)
                        sample = sample[0, :, :, :] / 255
                        if (self.input_channel == 1):
                            sample = sample[:, :, 0]
                            plt.imshow(sample, cmap='gray')
                        elif (self.input_channel == 3):
                            plt.imshow(sample)
                        plt.show()

                        ih = img_hat.detach().numpy()
                        shape = (self.BATCH_SIZE, self.input_width, self.input_height, self.input_channel)
                        sample = np.reshape(ih, shape)
                        sample = sample[0, :, :, :]
                        print(sample.shape)
                        if (self.input_channel == 1):
                            sample = sample[:, :, 0]
                            plt.imshow(sample, cmap='gray')
                        elif (self.input_channel == 3):
                            plt.imshow(sample)
                        plt.show()
                        print

            torch.save(self.net, './LDAMP saved models/customLDAMP' + str(self.n_train_images))
        elif (stage == 'testing'):
            with torch.no_grad():
                self.net = torch.load('./LDAMP saved models/customLDAMP75000')
                self.net.eval()
                history = []
                # val_psnrs = []
                for i, img in enumerate(self.testdataloader):
                    print("Processing batch " + str(i) + "...")
                    A_val = self.sensing_method.returnSensingMatrix()
                    A_val = A_val.detach()

                    shape = (self.BATCH_SIZE, self.input_channel, self.input_width * self.input_height)
                    img = np.reshape(img, shape)
                    img = img.type(torch.float).cuda()
                    measurement = self.sensing_method(img)
                    img_hat = self.net(measurement, torch.zeros(measurement.shape), A_val)

                    _, _, error = utils.EvalError(img_hat, img)
                    history.append(error)

                    # display the original image
                    sample = (np.reshape(img, (self.input_width, self.input_height)))
                    if (self.input_channel == 1):
                        plt.imshow(sample, cmap='gray')
                    elif (self.input_channel == 3):
                        plt.imshow(sample)
                    plt.show()

                    # display reconstructed image
                    ih = img_hat.detach().numpy()
                    ih = ih[:,0]
                    sample = np.reshape(ih, (self.specifics["input_width"], self.specifics["input_height"]))
                    plt.imshow(sample, cmap='gray')
                    plt.show()

                    # val_psnrs.append(utils.compute_average_psnr(img.cpu().double(), img_hat.detach().cpu().double()))
                # val_psnr = sum(val_psnrs) / len(val_psnrs)
                # print("average test psnr (higher is better): " + str(val_psnr))
                psnrs = sum(history) / len(history)
                print("average test psnr my own: " + str(psnrs))

class LearnedDAMP(nn.Module):
    def __init__(self, specifics):
        super(LearnedDAMP, self).__init__()
        self.name = 'LearnedDAMP'
        self.specifics = specifics
        self.n = self.specifics["n"]
        self.m = self.specifics["m"]
        self.BATCH_SIZE = self.specifics["BATCH_SIZE"]
        self.layers = nn.Sequential()
        for i in range(self.specifics["max_n_DAMP_layers"]):
            self.layers.add_module("Layer" + str(i),LearnedDAMPLayer(specifics))

    def forward(self, y_measured, A_val):
        z = y_measured
        xhat = torch.zeros([self.n, self.BATCH_SIZE], dtype=torch.float32)
        for i, l in enumerate(self.layers):
            print("\tLayer " + str(i))
            (next_xhat, next_z) = l(xhat, z, A_val, y_measured)
            z = next_z
            xhat = next_xhat
        return xhat


class LearnedDAMPLayer(nn.Module):
    def __init__(self, specifics):
        super(LearnedDAMPLayer, self).__init__()

        # instantiate carry-over variables
        self.specifics = specifics
        self.input_channel = specifics["input_channel"]
        self.input_width = specifics["input_width"]
        self.input_height = specifics["input_height"]
        self.BATCH_SIZE = specifics["BATCH_SIZE"]
        self.m = specifics["m"]
        self.n = specifics["n"]
        self.n_DAMP_layers = specifics["max_n_DAMP_layers"]
        self.DnCNN_wrapper = DnCNN_wrapper(specifics=specifics)

        # self.A_val = utils.generateAVal(self.m,self.n)
        # self.z = 0 # used in the LDAMP calculation between layers

        # self.MSE_history = []  # Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
        # self.NMSE_history = []
        # self.PSNR_history = []

    def forward(self, xhat, z, A_val, y_measured):
        z = z.cpu()
        # self.A_val = A_val
        # self.z = y_measured
        # xhat = torch.zeros([self.n, self.BATCH_SIZE], dtype=torch.float32)

        # sample = np.reshape(x_true.detach().cpu(), (self.specifics["input_width"], self.specifics["input_height"], self.specifics["BATCH_SIZE"]))
        # print(sample[:, :])

        # for i in range(self.n_DAMP_layers):  # self.layers, this is the number of AMP layers
        #     print("DAMP Layer " + str(i) + "...")
        r_addme = utils.At_handle(A_val, z)
        r = xhat + r_addme#torch.add(xhat, utils.At_handle(self.A_val,self.z))
        # r = torch.t(r)

        # sample = np.reshape(r.detach().cpu(), (self.specifics["input_width"], self.specifics["input_height"], self.specifics["BATCH_SIZE"]))
        # print(self.z.size())
        # print(self.z)

        (xhat, dxdr) = self.DnCNN_wrapper(r)
        # xhat = torch.t(xhat)
        z = y_measured - utils.A_handle(A_val, xhat).cuda()
        # print(dxdr.shape)
        # print(z.shape)
        # z_addme = (float(self.n) / float(self.m) * dxdr * z.cuda())
        # z = z.cuda() + z_addme
        # print(xhat)
        # (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = utils.EvalError(xhat, x_true)
        # # print(PSNR_thisiter)
        # self.MSE_history.append(MSE_thisiter)
        # self.NMSE_history.append(NMSE_thisiter)
        # self.PSNR_history.append(PSNR_thisiter)

        return xhat, z#, self.MSE_history, self.NMSE_history, self.PSNR_history


class DnCNN_wrapper(nn.Module):
    """
        Denoiser wrapper to compute divergence
    """

    def __init__(self, specifics):
        super(DnCNN_wrapper, self).__init__()
        self.DnCNN = DnCNN(specifics=specifics)

    def forward(self, r):
        xhat = self.DnCNN(r)
        # sample = np.reshape(xhat.detach().cpu(), (self.specifics["input_width"], self.specifics["input_height"], self.specifics["BATCH_SIZE"]))
        # print(sample[:, :])

        r_abs = torch.abs(r)
        r_absColumnWise = torch.max(r_abs, dim=0)  # gives a tuple ([max of each row] , [index of where it was found]) # reduce on dim=0 to get rid of n and leave BATCH_SIZE
        epsilon = torch.max(r_absColumnWise[0] * 0.001, torch.tensor(0.00001))  # of size [BATCH_SIZE]
        eta = torch.empty(r.shape).normal_(mean=0, std=1)
        r_perturbed = r + torch.mul(eta, epsilon)
        xhat_perturbed = self.DnCNN(r_perturbed.detach())
        # print(torch.mean(xhat_perturbed - xhat))
        eta_dx = torch.mul(eta, xhat_perturbed - xhat)  # Want element-wise multiplication
        mean_eta_dx = torch.mean(eta_dx, dim=0)
        # print(torch.mean(mean_eta_dx))
        dxdr = torch.div(mean_eta_dx, epsilon).cuda()
        # dxdr.detach()
        # print(torch.mean(dxdr))
        return xhat, dxdr

class DnCNN(nn.Module):
    def __init__(self, specifics):
        super(DnCNN, self).__init__()
        self.name = 'DnCNN'
        self.specifics = specifics
        self.channel_img = self.specifics["input_channel"]
        self.width_img = self.specifics["input_width"]
        self.height_img = self.specifics["input_height"]

        denoiser = nn.ModuleList()
        conv0 = nn.Conv2d(specifics["input_channel"], 64, kernel_size=3, stride=1, padding=1, bias=False)
        relu0 = nn.ReLU()
        denoiser.add_module("conv0", conv0)
        denoiser.add_module("relu0", relu0)
        for j in range(1, specifics["n_DnCNN_layers"] - 1):
            conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            batchnorm = nn.BatchNorm2d(64)  # training
            relu = nn.ReLU()
            denoiser.add_module("conv" + str(j), conv)
            denoiser.add_module("batch" + str(j), batchnorm)
            denoiser.add_module("relu" + str(j), relu)
        convLast = nn.Conv2d(64, specifics["input_channel"], kernel_size=3, stride=1, padding=1, bias=False)
        denoiser.add_module("convLast", convLast)
        self.dncnnLayers = denoiser

    def forward(self, r):
        orig_Shape = r.shape
        shape4D = [-1, self.channel_img, self.width_img, self.height_img]
        r = torch.reshape(r, shape4D)

        x = r
        for i, l in enumerate(self.dncnnLayers):
            x = l(x)

        x_hat = r - x
        x_hat = torch.reshape(x_hat, orig_Shape)
        return x_hat
