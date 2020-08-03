import torch
from torch import nn
import utils
import matplotlib.pyplot as plt
import numpy as np
import testNet
import scipy.linalg as la

if torch.cuda.is_available():
    print("Detected " + str(torch.cuda.device_count()) + " GPU(s)!")
    device = torch.device('cuda')
else:
    print("GPU unavailable, using CPU!")
    device = torch.device('cpu')


def reconstruction_method(reconstruction,specifics):
    if(reconstruction == 'LearnedDAMP' or reconstruction == 'LDAMP'):
        ldamp_wrap = LDAMP_wrapper(specifics)
        return ldamp_wrap

class LDAMP_wrapper():
    def __init__(self, specifics):
        self.name = 'LearnedDAMPwrap'
        self.specifics = specifics

    def initialize(self, dset,sensing_method,specifics):
        # set up variables
        self.BATCH_SIZE = specifics["BATCH_SIZE"]
        self.input_channel = specifics["input_channel"]
        self.input_width = specifics["input_width"]
        self.input_height = specifics["input_height"]
        self.n = specifics["n"]
        self.m = specifics["m"]
        self.n_train_images = self.specifics['n_train_images']
        self.n_val_images = self.specifics["n_val_images"]
        self.EPOCHS = self.specifics["EPOCHS"]
        self.previously_trained = self.specifics["previously_trained"]
        self.usersName = specifics["fileName"]

        # set up network and inputs
        self.dset = dset
        self.sensing_method = sensing_method
        self.A_val = self.sensing_method.returnSensingMatrix().requires_grad_(False).to(device)
        self.A_val_pinv = torch.Tensor(la.pinv(self.A_val.cpu())).requires_grad_(False).to(device)
        self.net = LearnedDAMP(specifics=specifics, A_val=self.A_val, A_val_pinv=self.A_val_pinv).to(device)
        if(specifics["resume"]):
            self.net.load_state_dict(torch.load(specifics["load_network"]))

        # self.net = testNet.testLearnedDAMP(specifics=specifics)
        self.dataloader = torch.utils.data.DataLoader(self.dset,
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)
        self.testdataloader=torch.utils.data.DataLoader(self.dset[:self.n_val_images],
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)
        self.loss_f = torch.nn.MSELoss(reduction='mean').to(device)  # use 'mean' for torch versions after 1.0.0

        params = list(self.net.parameters())
        print("Number of parameters: " + str(len(params)))

    def run(self, stage):
        if (stage == 'training'):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.specifics['learning_rate'])
            mseData = []
            psnrData = []
            for epoch in range(self.EPOCHS):
                print("\nBeginning EPOCH " + str(epoch) + "...")
                for i, img in enumerate(self.dataloader):
                    if(img.shape[0] % self.BATCH_SIZE != 0):
                        break;
                    # zero gradients
                    self.optimizer.zero_grad()
                    img = img.type(torch.float).to(device)
                    # img = img.reshape((self.BATCH_SIZE, self.input_channel * self.input_width * self.input_height))
                    measurement = self.sensing_method(img)

                    # forward function
                    img_hat = self.net(measurement)
                    img_hat = torch.reshape(img_hat, img.shape)

                    # calculate loss and backpropogate
                    loss = self.loss_f(img_hat, img)
                    loss.backward()
                    self.optimizer.step()

                    # print graphs error and graphs
                    if (i % 100 == 0):
                        currentNumImgs = (i + 1) * self.BATCH_SIZE + epoch * self.n_train_images + self.previously_trained
                        mse_error, psnr = utils.EvalError(img_hat, img)

                        print("Processed " + str(currentNumImgs) + " images"
                              + " MSE Loss: " + "%.4f" % mse_error.clone().cpu().detach_().numpy()
                              + " PSNR: " + "%.4f" % psnr.clone().cpu().detach_().numpy())
                        mseData.append(mse_error.clone().cpu().detach_().numpy())
                        psnrData.append(psnr.clone().cpu().detach_().numpy())
                print("**QUICKSAVE**")
                currentNumImgs = (epoch+1) * self.n_train_images + self.previously_trained
                torch.save(self.net.state_dict(), './LDAMP saved models/quickSave' + self.usersName + str(currentNumImgs))
            torch.save(self.net.state_dict(), './LDAMP saved models/completed' + self.usersName + str(self.n_train_images))
            plt.plot(mseData)
            plt.xlabel('Epoch')
            plt.ylabel('MSE loss')
            plt.show()
            plt.plot(psnrData)
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.show()

        elif (stage == 'testing'):
            with torch.no_grad():
                self.net.load_state_dict(torch.load(self.specifics["load_network"]))
                self.net.eval()
                history = []
                for i, img in enumerate(self.testdataloader):
                    print("Processing batch " + str(i) + "...")

                    img = img.type(torch.float).to(device)
                    measurement = self.sensing_method(img)

                    # forward function
                    img_hat = self.net(measurement)
                    img_hat = torch.reshape(img_hat, img.shape)

                    mse_error, psnr = utils.EvalError(img_hat, img)
                    history.append(psnr)
                psnrs = sum(history) / len(history)
                print("average test psnr my own: " + str(psnrs))

class LearnedDAMP(nn.Module):
    def __init__(self, specifics, A_val, A_val_pinv):
        super(LearnedDAMP, self).__init__()
        self.name = 'LearnedDAMP'
        self.specifics = specifics
        self.n = self.specifics["n"]
        self.m = self.specifics["m"]
        self.BATCH_SIZE = self.specifics["BATCH_SIZE"]
        self.A_val = A_val
        self.A_val_pinv = A_val_pinv
        self.layers = nn.Sequential().to(device)
        for i in range(self.specifics["max_n_DAMP_layers"]):
            self.layers.add_module("Layer" + str(i),LearnedDAMPLayer(specifics))

    def forward(self, y_measured):
        y_measured = y_measured.t()
        z = y_measured
        xhat = torch.zeros([self.n, self.BATCH_SIZE], dtype=torch.float32).to(device)
        for i, l in enumerate(self.layers):
            # print("\tDAMP Layer " + str(i))
            (next_xhat, next_z) = l(xhat, z, self.A_val, self.A_val_pinv, y_measured)
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
        self.DnCNN_wrapper = DnCNN_wrapper(specifics=specifics).to(device)

    def forward(self, xhat, z, A_val, A_val_pinv, y_measured):
        r = xhat + utils.A_handle(A_val_pinv, z) # TODO SIGNIFICANT CHANGE HERE
        (xhat, dxdr) = self.DnCNN_wrapper(r)

        z = y_measured - utils.A_handle(A_val, xhat).to(device)
        z = z + (float(1)/ float(self.m) * dxdr * z.to(device)) #TODO SIGNIFICANT CHANGE HERE
        return xhat, z

class DnCNN_wrapper(nn.Module):
    def __init__(self, specifics):
        super(DnCNN_wrapper, self).__init__()
        if torch.cuda.device_count() > 1:
            self.DnCNN = nn.DataParallel(DnCNN(specifics=specifics).to(device))
        else:
            self.DnCNN = DnCNN(specifics=specifics).to(device)

    def forward(self, r):
        xhat = self.DnCNN(r)

        r_abs = torch.abs(r)
        r_absColumnWise = torch.max(r_abs, dim=0)  # gives a tuple ([max of each row] , [index of where it was found]) # reduce on dim=0 to get rid of n and leave BATCH_SIZE
        epsilon = torch.max(r_absColumnWise[0] * 0.001, torch.tensor(0.00001).to(device))  # of size [BATCH_SIZE]
        eta = torch.empty(r.shape).normal_(mean=0, std=1).to(device)
        r_perturbed = r + torch.mul(eta, epsilon)
        xhat_perturbed = self.DnCNN(r_perturbed.detach())
        eta_dx = torch.mul(eta, xhat_perturbed - xhat)  # Want element-wise multiplication
        mean_eta_dx = torch.mean(eta_dx, dim=0)
        dxdr = torch.div(mean_eta_dx, epsilon).to(device)
        dxdr.detach() # turn off gradient calculation

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
        conv0 = nn.Conv2d(specifics["input_channel"], 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        relu0 = nn.ReLU().to(device)
        denoiser.add_module("conv0", conv0)
        denoiser.add_module("relu0", relu0)
        for j in range(1, specifics["n_DnCNN_layers"] - 1):
            conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
            batchnorm = nn.BatchNorm2d(64).to(device) # training
            relu = nn.ReLU().to(device)
            denoiser.add_module("conv" + str(j), conv)
            denoiser.add_module("batch" + str(j), batchnorm)
            denoiser.add_module("relu" + str(j), relu)
        convLast = nn.Conv2d(64, specifics["input_channel"], kernel_size=3, stride=1, padding=1, bias=False).to(device)
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