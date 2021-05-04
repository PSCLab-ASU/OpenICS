import ISTANet.sensing_methods as sensing_methods
import ISTANet.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
from time import time
from torch.nn import init
import cv2
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reconstruction_method(reconstruction, specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    ISTANet_model = ISTANet_wrapper(reconstruction, specifics)

    return ISTANet_model


class ISTANet_wrapper():
    def __init__(self,reconstruction, specifics):
        # do the initialization of the network with given parameters.
        self.specifics = specifics

        self.model = 0
        if(reconstruction == "ISTANetPlus"):
            model = ISTANetplus(self.specifics['layer_num'], specifics)
            model = nn.DataParallel(model)
            model = model.to(device)
        elif(reconstruction == "ISTANet"):
            model = ISTANet(self.specifics['layer_num'], specifics)
            model = nn.DataParallel(model)
            model = model.to(device)
        self.model = model

    def initialize(self,dataset, TrainingLabels, sensing):
        # do the preparation for the running.
        self.rand_loader = dataset
        self.trainingLabels = TrainingLabels

    def run(self):
        if (self.specifics['stage'] == 'training'):
            model = self.model

            start_epoch = self.specifics['start_epoch']
            end_epoch = self.specifics['end_epoch']
            learning_rate = self.specifics['learning_rate']
            layer_num = self.specifics['layer_num']
            group_num = self.specifics['group_num']
            cs_ratio = self.specifics['cs_ratio']
            batch_size = self.specifics['batch_size']
            nrtrain = self.specifics['nrtrain']
            model_dir = self.specifics['model_dir']
            log_dir = self.specifics['log_dir']

            print_flag = 1  # print parameter number

            if print_flag:
                num_count = 0
                for para in model.parameters():
                    num_count += 1
                    print('Layer %d' % num_count)
                    print(para.size())

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model_dir = "%s/CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f_imgwidth_%d" % (
            model_dir, layer_num, group_num, cs_ratio, learning_rate, self.specifics['input_width'])

            log_file_name = "%s/Log_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f_imgwidth_%d.txt" % (
            log_dir, layer_num, group_num, cs_ratio, learning_rate, self.specifics['input_width'])

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            if start_epoch > 0:
                pre_model_dir = model_dir
                model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

            # added functionality
            # if(self.specifics['bmp_folder_images'] == True):
            #     Training_labels = utils.getgetTrainingLabelsFolder(self.specifics['stage'], self.specifics)
            #     Phi_input, Qinit = sensing_methods.computInitMxScratch(Training_labels=Training_labels,
            #                                                            specifics=self.specifics)
            # else:
            # input_channel = self.specifics['input_channel']
            # input_width = self.specifics['input_width']
            # input_height = self.specifics['input_width']
            Training_labels = self.trainingLabels
            Phi_input, Qinit = sensing_methods.computInitMx(Training_labels=Training_labels,
                                                            specifics=self.specifics)



            Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
            Phi = Phi.to(device)

            Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
            Qinit = Qinit.to(device)

            # Training loop
            print('Training on GPU? ' + str(torch.cuda.is_available()))
            for epoch_i in range(start_epoch + 1, end_epoch + 1):
                for j, data in enumerate(self.rand_loader):

                    batch_x = data
                    batch_x = batch_x.to(device)

                    Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

                    [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)

                    # Compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                    for k in range(layer_num - 1):
                        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))

                    gamma = torch.Tensor([0.01]).to(device)

                    # loss_all = loss_discrepancy
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss_all.backward()
                    optimizer.step()

                    output_data = "[%02d/%02d][%d/%d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (
                    epoch_i, end_epoch, j * batch_size, nrtrain, loss_all.item(), loss_discrepancy.item(),
                    loss_constraint)
                    print(output_data)

                if (not (os.path.exists("" + log_dir))):
                    os.mkdir("" + log_dir)
                output_file = open(log_file_name, 'a')
                output_file.write(output_data)
                output_file.close()

                if epoch_i % 5 == 0:
                    torch.save(model.state_dict(),
                               "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
        elif (self.specifics['stage'] == 'testing'):
            model = self.model

            epoch_num = self.specifics['testing_epoch_num']
            learning_rate = self.specifics['learning_rate']
            layer_num = self.specifics['layer_num']
            group_num = self.specifics['group_num']
            cs_ratio = self.specifics['cs_ratio']
            model_dir = self.specifics['model_dir']
            log_dir = self.specifics['log_dir']
            data_dir = self.specifics['data_dir']
            result_dir = self.specifics['result_dir']
            test_name = self.specifics['Testing_data_location']

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            model_dir = "%s/CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f_imgwidth_%d" % (
            model_dir, layer_num, group_num, cs_ratio, learning_rate, self.specifics['input_width'])

            # Load pre-trained model with epoch number
            model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))


            if(self.specifics['testing_data_isFolderImages'] == False):
                raise Exception('testing_data_isFolderImages must be True (Data must be bmp or tif in folder)')

            test_dir = os.path.join(data_dir, test_name)
            if(self.specifics['testing_data_type'] == 'tif'):
                filepaths = glob.glob(test_dir + '/*.tif')
            elif(self.specifics['testing_data_type'] == 'bmp'):
                filepaths = glob.glob(test_dir + '/*.bmp')
            elif (self.specifics['testing_data_type'] == 'png'):
                filepaths = glob.glob(test_dir + '/*.png')
            elif (self.specifics['testing_data_type'] == 'jpg'):
                filepaths = glob.glob(test_dir + '/*.jpg')
            else:
                raise Exception('testing_data_type of ' + self.specifics['testing_data_type'] + ' is unknown')

            result_dir = os.path.join(result_dir, test_name)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            ImgNum = len(filepaths)
            REC_TIME_All = np.zeros([1, ImgNum], dtype=np.float32)
            PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
            SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

            # input_channel = self.specifics['input_channel']
            # input_width = self.specifics['input_width']
            # input_height = self.specifics['input_width']
            Training_labels = self.trainingLabels
            Phi_input, Qinit = sensing_methods.computInitMx(Training_labels=Training_labels, specifics=self.specifics)

            Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
            Phi = Phi.to(device)

            Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
            Qinit = Qinit.to(device)

            print('\n')
            print("CS Reconstruction Start")
            with torch.no_grad():
                for img_no in range(ImgNum):
                    imgName = filepaths[img_no]

                    Img = cv2.imread(imgName, 1)

                    #Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                    Img_yuv = Img
                    Img_rec_yuv = Img_yuv.copy()

                    if (self.specifics['sudo_rgb']):
                        start = time()
                        x_output3 = []
                        Iorg_3 = []
                        for i in range(3):
                            Iorg_y = Img_yuv[:, :, i]
                            [Iorg, row, col, Ipad, row_new, col_new] = utils.imread_CS_py(Iorg_y, self.specifics)
                            Iorg_3.append(Iorg)
                            Icol = utils.img2col_py(Ipad, self.specifics['input_width']).transpose() / 255.0

                            Img_output = Icol

                            batch_x = torch.from_numpy(Img_output)
                            batch_x = batch_x.type(torch.FloatTensor)
                            batch_x = batch_x.to(device)

                            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

                            [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
                            x_output3.append(x_output)

                        x_output = x_output3
                        Iorg = np.array(Iorg_3)
                        end = time()
                        Pred_t = []
                        for i in range(3):
                            Pred_t.append(x_output[i].cpu().data.numpy().transpose())
                    else:
                        Iorg_y = Img_yuv[:, :, 0]
                        [Iorg, row, col, Ipad, row_new, col_new] = utils.imread_CS_py(Iorg_y, self.specifics)
                        Icol = utils.img2col_py(Ipad, self.specifics['input_width']).transpose() / 255.0

                        Img_output = Icol

                        start = time()
                        batch_x = torch.from_numpy(Img_output)
                        batch_x = batch_x.type(torch.FloatTensor)
                        batch_x = batch_x.to(device)
                        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

                        [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
                        end = time()
                        Prediction_value = x_output.cpu().data.numpy()
                        Pred_t = Prediction_value.transpose()

                    X_rec = np.clip(utils.col2im_CS_py(Pred_t, row, col, row_new, col_new, self.specifics), 0, 1)

                    rec_PSNR = utils.psnr(X_rec * 255, Iorg.astype(np.float64))
                    if(self.specifics['sudo_rgb']):
                        rec_SSIM = ssim(X_rec.transpose() * 255, Iorg.astype(np.float64).transpose(), data_range=255, multichannel=True)
                    else:
                        rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

                    # plt.imshow(X_rec, cmap='gray_r')
                    # plt.show()
                    print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                    img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

                    # TODO change: no longer save the images
                    # Img_rec_yuv[:, :, 0] = X_rec * 255

                    # im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                    # im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    # resultName = imgName.replace(data_dir, result_dir)
                    # cv2.imwrite("%s_ISTA_Net_plus_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
                    # resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)

                    del x_output
                    REC_TIME_All[0, img_no] = (end-start)
                    PSNR_All[0, img_no] = rec_PSNR
                    SSIM_All[0, img_no] = rec_SSIM

            print('\n')
            output_data = "CS ratio is %d, Avg TIME/PSNR/SSIM for %s is %.4f/%.2f/%.4f, Epoch number of model is %d \n" % (
            cs_ratio, test_name, np.mean(REC_TIME_All), np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
            print(output_data)

            output_file_name = "%s/PSNR_SSIM_Results_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f_imgwidth_%d.txt" % (
            log_dir, layer_num, group_num, cs_ratio, learning_rate, self.specifics['input_width'])

            if (not (os.path.exists("" + log_dir))):
                os.mkdir("" + log_dir)
            output_file = open(output_file_name, 'a')
            output_file.write(output_data)
            output_file.close()

            print("CS Reconstruction End")

# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, specifics):
        super(BasicBlock, self).__init__()
        self.specifics = specifics

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        # 3 x 3 x Nf (Nf = 32 is default)
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, self.specifics['input_width'], self.specifics['input_width'])

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x_pred = x_pred.view(-1, self.specifics['input_width']*self.specifics['input_width'])

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo, specifics):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock(specifics))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]

# Define ISTA-Net Block
class BasicBlock2(torch.nn.Module):
    def __init__(self, specifics):
        super(BasicBlock2, self).__init__()
        self.specifics = specifics

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        # 3 x 3 x Nf (Nf = 32 is default)
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, self.specifics['input_width'], self.specifics['input_width'])

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, self.specifics['input_width']*self.specifics['input_width'])

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]

# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo, specifics):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock2(specifics))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
