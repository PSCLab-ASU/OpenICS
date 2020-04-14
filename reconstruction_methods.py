import numpy as np
import scipy.io as sio
import torch
from torch import nn
import cv2
from time import time
import glob
from istanet import ISTANetModel
from skimage.measure import compare_ssim as ssim
from utils import rgb2ycbcr, ycbcr2rgb, imread_CS_py, img2col_py, col2im_CS_py, psnr


def reconstruction_method(reconstruction, specifics):
    # a function to return the reconstruction method with given parameters.
    # It's a class with two methods: initialize and run.
    if reconstruction == "ISTANet":
        return ISTANet(specifics)


class ISTANet:
    def __init__(self, specifics):
        # do the initialization of the network with given parameters.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.specifics = specifics
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.sensing = None
        self.Phi = None
        self.Qinit = None

    # def initialize(self, dataset, sensing):
    def initialize(self, sensing):
        # do the preparation for the running.
        # init optimizers if training
        # testing is one time forward
        # self.dataset = dataset
        self.sensing = sensing

        self.model = ISTANetModel(self.specifics["layer_num"])
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.specifics["learning_rate"]
        )

        # Phi is their sampling matrix, should this be involved in the sensing methods section instead?
        Phi_data = sio.loadmat(f"models/phi_0_{self.specifics['cs_ratio']}_1089.mat")
        Phi_input = Phi_data["phi"]
        Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
        self.Phi = Phi.to(self.device)
        # print(self.Phi.size())

        # setup qinit
        Training_data_Name = "Training_Data.mat"
        Training_data = sio.loadmat("models/Training_Data.mat")
        Training_labels = Training_data["labels"]

        X_data = Training_labels.transpose()
        Y_data = np.dot(Phi_input, X_data)
        Y_YT = np.dot(Y_data, Y_data.transpose())
        X_YT = np.dot(X_data, Y_data.transpose())
        self.Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
        self.Qinit = torch.from_numpy(self.Qinit).type(torch.FloatTensor)
        self.Qinit = self.Qinit.to(self.device)

    def run(self, stage):
        if stage == "training":
            start_epoch = self.specifics["start_epoch"]
            end_epoch = self.specifics["end_epoch"]
            if start_epoch > 0:
                self.model.load_state_dict(
                    torch.load(f"models/istanet_params_{start_epoch}.pkl")
                )

            for epoch in range(start_epoch + 1, end_epoch + 1):
                for data in self.sensing:
                    batch_x = data
                    batch_x = batch_x.to(self.device)
                    # print("PHI SIZE AT MULT:", self.Phi.size())
                    Phix = torch.mm(batch_x, torch.transpose(self.Phi, 0, 1))
                    # print(
                    #     "PHIX AT CREATION",
                    #     Phix.size(),
                    #     "PHI SIZE AFTER MULT:",
                    #     self.Phi.size(),
                    # )
                    [x_output, loss_layers_sym] = self.model.forward(
                        Phix, self.Phi, self.Qinit
                    )

                    # Compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                    for k in range(self.specifics["layer_num"] - 1):
                        loss_constraint += torch.mean(
                            torch.pow(loss_layers_sym[k + 1], 2)
                        )

                    gamma = torch.Tensor([0.01]).to(self.device)

                    # loss_all = loss_discrepancy
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                    # Zero gradients, perform a backward pass, and update the weights.
                    self.optimizer.zero_grad()
                    loss_all.backward()
                    self.optimizer.step()

                    output_data = (
                        "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n"
                        % (
                            epoch,
                            end_epoch,
                            loss_all.item(),
                            loss_discrepancy.item(),
                            loss_constraint,
                        )
                    )
                    print(output_data)

                    if epoch % 5 == 0:
                        torch.save(
                            self.model.state_dict(), f"models/net_params_{epoch}.pkl"
                        )  # save only the parameters
        elif stage == "testing":
            print(self.specifics)
            self.model.load_state_dict(
                torch.load(
                    f"models/net_params_{self.specifics['epoch_num']}.pkl"
                )
            )

            print("CS Reconstruction Start")
            filepaths = glob.glob(f"data/{self.dataset}/*.tif")

            result_dir = "results"

            ImgNum = len(filepaths)
            PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

            with torch.no_grad():
                for img_no in range(self.specifics["test_image_number"]):
                    imgName = filepaths[img_no]

                    Img = cv2.imread(imgName, 1)

                    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()

                    Iorg_y = Img_yuv[:, :, 0]

                    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                    Icol = img2col_py(Ipad, 33).transpose() / 255.0

                    Img_output = Icol

                    start = time()

                    batch_x = torch.from_numpy(Img_output)
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_x = batch_x.to(self.device)

                    Phix = torch.mm(batch_x, torch.transpose(self.Phi, 0, 1))

                    [x_output, loss_layers_sym] = self.model(Phix, self.Phi, self.Qinit)

                    end = time()

                    Prediction_value = x_output.cpu().data.numpy()

                    # loss_sym = torch.mean(torch.pow(loss_layers_sym[0], 2))
                    # for k in range(layer_num - 1):
                    #     loss_sym += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))
                    #
                    # loss_sym = loss_sym.cpu().data.numpy()

                    X_rec = np.clip(
                        col2im_CS_py(
                            Prediction_value.transpose(), row, col, row_new, col_new
                        ),
                        0,
                        1,
                    )

                    rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
                    rec_SSIM = ssim(
                        X_rec * 255, Iorg.astype(np.float64), data_range=255
                    )

                    print(
                        "[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f"
                        % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM)
                    )

                    Img_rec_yuv[:, :, 0] = X_rec * 255

                    im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    resultName = imgName.replace("data", "output")
                    cv2.imwrite(
                        "%s_ISTA_Net_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png"
                        % (
                            resultName,
                            self.specifics["cs_ratio"],
                            self.specifics["epoch_num"],
                            rec_PSNR,
                            rec_SSIM,
                        ),
                        im_rec_rgb,
                    )
                    del x_output

                    PSNR_All[0, img_no] = rec_PSNR
                    SSIM_All[0, img_no] = rec_SSIM

            print("\n")
            output_data = (
                "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n"
                % (
                    self.specifics["cs_ratio"],
                    self.dataset,
                    np.mean(PSNR_All),
                    np.mean(SSIM_All),
                    self.specifics["epoch_num"],
                )
            )
            print(output_data)

            print("CS Reconstruction End")
