import torch
from torch import nn
import torch.nn.functional as F
import utils


def reconstruction_method(reconstruction,specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if(reconstruction == 'LearnedDAMP' or reconstruction == 'LDAMP'):
        ldamp_wrap = LDAMP_wrapper(specifics)
        return ldamp_wrap

class LDAMP_wrapper():
    def __init__(self, specifics):
        self.name = 'LearnedDAMP'
        self.specifics = specifics

    def initialize(self, dset,sensing_method,specifics):
        # shared between both training and testing
        self.dset = dset
        self.sensing_method = sensing_method
        self.net = LearnedDAMP(specifics)
        self.dataloader = torch.utils.data.DataLoader(self.dset,
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)
        self.testdataloader=torch.utils.data.DataLoader(self.dset,
                                                  batch_size=self.specifics["BATCH_SIZE"], shuffle=True, num_workers=2)

    def run(self, stage):
        if (stage == 'training'):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.specifics['learning_rate'])
            self.loss_f = torch.nn.MSELoss(reduction='mean')
            for epoch in range(self.specifics["EPOCHS"]):
                count = 0
                for img in self.dataloader:
                    print("Processing batch " + str(count) + "...")
                    img = img.cuda()
                    self.optimizer.zero_grad()
                    measurement = self.sensing_method(img)
                    img_hat = self.net(measurement)
                    loss = self.loss_f(img_hat, img)
                    loss.backward()
                    self.optimizer.step()
                    count += 1
            # torch.save(LDAMP, './savedModels/ldamp1')

        elif (stage == 'testing'):
            # self.net = torch.load('./savedModels/ldamp1') # load the model you want to test
            with torch.no_grad():
                self.net.eval()
                val_psnrs = []
                count = 0
                for img in self.testdataloader:
                    print("Processing batch " + str(count) + "...")
                    img = img.cuda()
                    measurement = self.sensing_method(img)
                    img_hat = self.net(measurement)
                    val_psnrs.append(utils.compute_average_psnr(img.cpu(), img_hat.detach().cpu()))
                    count += 1
                val_psnr = sum(val_psnrs) / len(val_psnrs)
                print("average test psnr (higher is better): " + str(val_psnr))

class LearnedDAMP(nn.Module):
    def __init__(self, specifics):
        super(LearnedDAMP, self).__init__()

        # instantiate carry-over variables
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

        def dncnnForward(r):
            channel_img = specifics["input_channel"]
            width_img = specifics["input_width"]
            height_img = specifics["input_height"]

            r = torch.t(r)
            orig_Shape = r.shape
            shape4D = [-1, channel_img, width_img, height_img]  # shape4D = [-1, height_img, width_img, channel_img]
            r = torch.reshape(r, shape4D)
            x = F.relu(self.conv0(r))
            for i, l in enumerate(self.layers):
                x = l(x)
            x = F.relu(self.convLast(x))
            x_hat = r - x
            x_hat = torch.t(torch.reshape(x_hat, orig_Shape))
            return x_hat
        self.dncnnForward = dncnnForward

    def forward(self, y_measured):
        self.z = y_measured
        xhat = torch.zeros([self.n, self.BATCH_SIZE], dtype=torch.float32)
        for iter in range(self.n_DAMP_layers):  # n_DAMP_layers should be 10, this is the number of denoiser layers
            print("Layer " + str(iter) + "...")
            r = torch.add(xhat, utils.At_handle(self.A_val,self.z))
            xhat = self.dncnnForward(r)

            r_abs = torch.abs(r)
            r_absColumnWise = torch.max(r_abs, 0)
            epsilon = torch.max(r_absColumnWise[0] * (torch.ones(self.BATCH_SIZE) * 0.001), torch.tensor(0.00001))
            eta = torch.empty(r.shape).normal_(mean=0,std=1)
            r_perturbed = r + torch.mul(eta, epsilon)
            xhat_perturbed = self.dncnnForward(r_perturbed)
            eta_dx = torch.mul(eta, xhat_perturbed - xhat)  # Want element-wise multiplication
            mean_eta_dx = torch.mean(eta_dx, dim=0)
            dxdr = torch.div(mean_eta_dx, epsilon).cuda()

            self.z = y_measured - utils.A_handle(self.A_val, xhat).cuda()
            self.z = self.z + (float(self.n) / float(self.m) * dxdr * self.z)
        return xhat
