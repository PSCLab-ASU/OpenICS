import torch
import torchvision
import LAPRAN.lapgan_adaptiveCS_resnet as lapgan
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
import os
import skimage 
from numpy.random import randn
import torchvision.utils as vutils
import torch.optim as optim
import LAPRAN.utils as utils
import math
import time
def reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n, specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    rec_method = None
    if (reconstruction == "LAPRAN"):
        rec_method = lapran(input_channel, input_width, input_height, m, n, specifics)
    return rec_method

class lapran():
    def __init__(self,input_channel, input_width, input_height, m, n, specifics):
        # do the initialization of the network with given parameters.

        torch.cuda.set_device(specifics["gpu"])
        print('Current gpu device: gpu %d' % (torch.cuda.current_device()))

        if specifics["seed"] is None:
            specifics["seed"] = np.random.randint(1, 10000)
        print('Random seed: ', specifics["seed"])
        np.random.seed(specifics["seed"])
        torch.manual_seed(specifics["seed"])
        if specifics["cuda"]:
            torch.cuda.manual_seed(specifics["seed"])
       # if not os.path.exists('%s/%s/cr%s/%s/test' % (specifics.outf, specifics.dataset, specifics.cr, specifics.model)):
        #    os.makedirs('%s/%s/cr%s/%s/test' % (specifics.outf, specifics.dataset, specifics.cr, specifics.model))
        self.criterion_mse = nn.MSELoss()
        self.channels = input_channel
        self.n = n

        self.batch_size = specifics["batch_size"]
        self.input_width = input_width
        self.input_height = input_height

        #For training 32x32 images, only the first 3 layers are used.
        self.cr4 = n//m
        if (input_width == 64):
            self.cr3 = 2*self.cr4
            self.cr2 = 4*self.cr4
            self.cr1 = 8*self.cr4
        elif(input_width==32):
            self.cr3 = self.cr4
            self.cr2 = 2*self.cr4
            self.cr1 = 4*self.cr4
        
        self.m4 = n // self.cr4
        self.m3 = n // self.cr3
        self.m2 = n // self.cr2
        self.m1 = n // self.cr1

        self.img_size4 = 64#input_width
        self.img_size3 = 32#input_width // 2
        self.img_size2 = 16#input_width // 4
        self.img_size1 = 8#input_width // 8

        
        cudnn.benchmark = True
        # For non-square image support in the future
        # self.img_width1 = input_width//8
        # self.img_width2 = input_width//4
        # self.img_width3 = input_width//2F
        # self.img_width4 = input_width

        # self.img_height1 = input_height//8
        # self.img_height2 = input_height//4
        # self.img_height3 = input_height//2
        # self.img_height4 = input_height
    def initialize(self,dataset,sensing,stage, specifics):
        if os.path.exists('sensing_matrix_cr%d_input_%dx%dx%d.npy' % (self.cr4,self.channels,self.input_width,self.input_height)):
            sensing_matrix4_np= np.load('sensing_matrix_cr%d_input_%dx%dx%d.npy' % (self.cr4,self.channels,self.input_width,self.input_height))
        else:
            sensing_matrix4_np = randn(self.channels, self.m4, self.n)
            np.save('sensing_matrix_cr%d_input_%dx%dx%d.npy' % (self.cr4,self.channels,self.input_width,self.input_height), sensing_matrix4_np)
        with torch.no_grad():
            self.sensing_matrix4 = sensing(self.n,self.m4,self.img_size4,self.channels, sensing_matrix4_np)
            self.sensing_matrix3 = sensing(self.n,self.m3,self.img_size4,self.channels, sensing_matrix4_np)
            self.sensing_matrix2 = sensing(self.n,self.m2,self.img_size4,self.channels, sensing_matrix4_np)
            self.sensing_matrix1 = sensing(self.n,self.m1,self.img_size4,self.channels, sensing_matrix4_np)


        self.g1_input = torch.FloatTensor(self.batch_size, self.channels, self.m1)
        self.g2_input = torch.FloatTensor(self.batch_size, self.channels, self.m2)
        self.g3_input = torch.FloatTensor(self.batch_size, self.channels, self.m3)
        self.g4_input = torch.FloatTensor(self.batch_size, self.channels, self.m4)

        self.g1_target = torch.FloatTensor(self.batch_size, self.channels, self.img_size1, self.img_size1)
        self.g2_target = torch.FloatTensor(self.batch_size, self.channels, self.img_size2, self.img_size2)
        self.g3_target = torch.FloatTensor(self.batch_size, self.channels, self.img_size3, self.img_size3)
        self.g4_target = torch.FloatTensor(self.batch_size, self.channels, self.img_size4, self.img_size4)

        self.y2 = torch.FloatTensor(self.batch_size, self.channels, self.m2)
        self.y3 = torch.FloatTensor(self.batch_size, self.channels, self.m3)
        self.y4 = torch.FloatTensor(self.batch_size, self.channels, self.m4)

        self.lapnet1_gen = lapgan.LAPGAN_Generator_level1(self.channels, self.channels * self.m1, specifics["ngpu"])
        self.lapnet1_disc = lapgan.LAPGAN_Discriminator_level1(self.channels, specifics["ngpu"])
        self.lapnet2_gen = lapgan.LAPGAN_Generator_level2(self.channels, self.channels * self.m2, specifics["ngpu"])
        self.lapnet2_disc = lapgan.LAPGAN_Discriminator_level2(self.channels, specifics["ngpu"])
        self.lapnet3_gen = lapgan.LAPGAN_Generator_level3(self.channels, self.channels * self.m3, specifics["ngpu"])
        self.lapnet3_disc = lapgan.LAPGAN_Discriminator_level3(self.channels, specifics["ngpu"])
        self.lapnet4_gen = lapgan.LAPGAN_Generator_level4(self.channels, self.channels * self.m4, specifics["ngpu"])
        self.lapnet4_disc = lapgan.LAPGAN_Discriminator_level4(self.channels, specifics["ngpu"])

      

        if specifics["cuda"]:
            self.lapnet1_gen.cuda(), self.lapnet1_disc.cuda()
            self.lapnet2_gen.cuda(), self.lapnet2_disc.cuda()
            self.lapnet3_gen.cuda(), self.lapnet3_disc.cuda()
            self.lapnet4_gen.cuda(), self.lapnet4_disc.cuda()

            self.criterion_mse.cuda()

            self.g1_input, self.g2_input, self.g3_input, self.g4_input = self.g1_input.cuda(), self.g2_input.cuda(), self.g3_input.cuda(), self.g4_input.cuda()
            self.g1_target, self.g2_target, self.g3_target, self.g4_target = self.g1_target.cuda(), self.g2_target.cuda(), self.g3_target.cuda(), self.g4_target.cuda()
            self.y2, self.y3, self.y4 = self.y2.cuda(), self.y3.cuda(), self.y4.cuda()
        '''
        if specifics["dataset"] == 'bsd500_patch':
            if 'wy' in specifics["model"] and 'ifusion' in specifics["model"]:
                if self.cr == 5:
                    self.level1_iter = 7 # 0.0264
                    self.level2_iter = 11 # 0.0157
                    self.level3_iter = 12 # 0.0090
                    self.level4_iter = 1
                elif self.cr == 10:
                    self.level1_iter = 16 # 0.0353 18  # 14
                    self.level2_iter = 16 # 0.0230 16  # 27
                    self.level3_iter = 16 # 0.0144 8  # 27
                    self.level4_iter = 21
                elif self.cr == 20:
                    self.level1_iter = 6 # 0.0450
                    self.level2_iter = 2 # 0.0389
                    self.level3_iter = 1 # 0.0318
                    self.level4_iter = 93
                elif self.cr == 30:
                    self.level1_iter = 5 # 0.0508
                    self.level2_iter = 0 # 0.0420 5
                    self.level3_iter = 0 # 0.0339 3

        if specifics["dataset"] == 'mnist':
            if 'wy' in specifics["model"] and 'ifusion' in specifics["model"]:
                if self.cr == 5:
                    self.level1_iter = 94 # 0.0074
                    self.level2_iter = 36 # 0.0052
                    self.level3_iter = 88
                    self.level4_iter = 1
                elif self.cr == 10:
                    self.level1_iter = 73 # 0.0193
                    self.level2_iter = 31 # 0.0111 16  # 27
                    self.level3_iter = 78 # 0.0077 8  # 27
                    self.level4_iter = 96 # 0.0034
                elif self.cr == 20:
                    self.level1_iter = 57 # 0.0499
                    self.level2_iter = 57 # 0.0233
                    self.level3_iter = 95 # 0.0122
                    self.level4_iter = 83 # 0.0059
                elif self.cr == 30:
                    self.level1_iter = 34 # 0.0771
                    self.level2_iter = 66 # 0.0378 5
                    self.level3_iter = 99 # 0.0177 3
                    self.level4_iter = 88  # 0.0056

        if specifics["dataset"] == 'cifar10':
            if 'wy' in specifics["model"] and 'ifusion' in specifics["model"]:
                if self.cr4 == 5:
                    self.level1_iter = 28  # 0.0126
                    self.level2_iter = 81  # 0.0039
                    self.level3_iter = 21  # 0.0025
                    self.level4_iter = 26  # 0.0008
                elif self.cr4 == 10:
                    self.level1_iter = 24  # 0.0242
                    self.level2_iter = 33  # 0.0102
                    self.level3_iter = 58  # 0.0045
                    self.level4_iter = 21  # 0.0017
                elif self.cr4 == 20:
                    self.level1_iter = 19  # 0.0420
                    self.level2_iter = 30  # 0.0221
                    self.level3_iter = 44  # 0.0111
                    self.level4_iter = 21  # 0.0043
                elif self.cr4 == 30:
                    self.level1_iter = 16  # 0.0532
                    self.level2_iter = 34  # 0.0317
                    self.level3_iter = 84  # 0.0174
                    self.level4_iter = 16  # 0.0078
            elif 'wy' in specifics["model"] and 'lfusion' in specifics["model"]:
                if self.cr4 == 10:
                    self.level1_iter = 18
                    self.level2_iter = 16
                    self.level3_iter = 22
                    self.level4_iter = 99
            elif 'woy' in specifics["model"]:
                if self.cr4 == 10:
                    self.level1_iter = 18 # 0.0251
                    self.level2_iter = 7  # 0.0376
                    self.level3_iter = 5  # 0.0374
                    self.level4_iter = 1  # 0.0388
            '''

        if (stage == "testing"):
            kwopt = {'num_workers': 1, 'pin_memory': True} if specifics["cuda"] else {}
            self.testloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, **kwopt)


            stage1_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1)
            stage2_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2)
            stage3_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3)
            stage4_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4)

            stage1_checkpoint = torch.load(stage1_path+"/best_checkpoint.pth")
            stage2_checkpoint = torch.load(stage2_path+"/best_checkpoint.pth")
            stage3_checkpoint = torch.load(stage3_path+"/best_checkpoint.pth")
            
            self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])
            self.lapnet2_gen.load_state_dict(stage2_checkpoint["generator_state_dict"])
            self.lapnet3_gen.load_state_dict(stage3_checkpoint["generator_state_dict"])
            if (self.input_width ==64):
                stage4_checkpoint = torch.load(stage4_path+"/best_checkpoint.pth")
                self.lapnet4_gen.load_state_dict(stage4_checkpoint["generator_state_dict"])

            self.lapnet1_gen.eval(), self.lapnet1_disc.eval()
            self.lapnet2_gen.eval(), self.lapnet2_disc.eval()
            self.lapnet3_gen.eval(), self.lapnet3_disc.eval()
            self.lapnet4_gen.eval(), self.lapnet4_disc.eval()
        elif (stage == "training"):
            self.best_iter = 0
            self.best_recons_loss = math.inf
            self.iter = 0
            if not os.path.exists('%s/%s/cr%s/%s/stage%s/model' % (specifics["outf"], specifics["dataset"], self.cr4,specifics["model"], specifics["stage"])):
                os.makedirs('%s/%s/cr%s/%s/stage%s/model' % (specifics["outf"], specifics["dataset"], self.cr4,specifics["model"], specifics["stage"]))
            if not os.path.exists('%s/%s/cr%s/%s/stage%s/image' % (specifics["outf"], specifics["dataset"], self.cr4,specifics["model"], specifics["stage"])):
                os.makedirs('%s/%s/cr%s/%s/stage%s/image' % (specifics["outf"], specifics["dataset"], self.cr4,specifics["model"], specifics["stage"]))
            kwopt = {'num_workers': 2, 'pin_memory': True} if specifics["cuda"] else {}
            self.trainloader = torch.utils.data.DataLoader(dataset[0], batch_size=self.batch_size, shuffle=True, **kwopt)
            self.valloader = torch.utils.data.DataLoader(dataset[1], batch_size=self.batch_size, shuffle=True, **kwopt)

            self.label = torch.FloatTensor(specifics["batch_size"])

            self.fake_label = 0.1
            self.real_label = 0.9

            # Weight initialization
            self.weights_init(self.lapnet1_gen,  init_type='normal'), self.weights_init(self.lapnet1_disc,  init_type='normal')
            self.weights_init(self.lapnet2_gen,  init_type='normal'), self.weights_init(self.lapnet2_disc, init_type='normal')
            self.weights_init(self.lapnet3_gen, init_type='normal'), self.weights_init(self.lapnet3_disc, init_type='normal')
            self.weights_init(self.lapnet4_gen, init_type='normal'), self.weights_init(self.lapnet4_disc, init_type='normal')



            self.optimizer_lapnet1_gen = optim.Adam(self.lapnet1_gen.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))
            self.optimizer_lapnet1_disc = optim.Adam(self.lapnet1_disc.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))

            self.optimizer_lapnet2_gen = optim.Adam(self.lapnet2_gen.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))
            self.optimizer_lapnet2_disc = optim.Adam(self.lapnet2_disc.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))

            self.optimizer_lapnet3_gen = optim.Adam(self.lapnet3_gen.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))
            self.optimizer_lapnet3_disc = optim.Adam(self.lapnet3_disc.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))

            self.optimizer_lapnet4_gen = optim.Adam(self.lapnet4_gen.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))
            self.optimizer_lapnet4_disc = optim.Adam(self.lapnet4_disc.parameters(), lr=specifics["lr"], betas=(0.5, 0.999))

            self.criterion_bce = nn.BCELoss()

            if specifics["cuda"]:
                self.criterion_bce.cuda()
                self.label = self.label.cuda()
            stage1_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1)
            stage2_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2)
            stage3_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3)
            stage4_path = '%s/%s/cr%s/%s/stage%s/model' % (
                specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4)
            if specifics["stage"] == 1:
                if os.path.exists(stage1_path+"/last_checkpoint.pth"):
                    stage1_checkpoint = torch.load(stage1_path+"/last_checkpoint.pth")
                    self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])
                    self.lapnet1_disc.load_state_dict(stage1_checkpoint["discriminator_state_dict"])
                    self.optimizer_lapnet1_gen.load_state_dict(stage1_checkpoint["gen_optim_state_dict"])
                    self.optimizer_lapnet1_disc.load_state_dict(stage1_checkpoint["disc_optim_state_dict"])

                    self.iter = stage1_checkpoint["iter"]
                    self.best_iter = stage1_checkpoint["best_iter"]
                    self.best_recons_loss = stage1_checkpoint["best_recons_loss"]
                    print('loading level1 iteration' + str(self.iter))
            if specifics["stage"] == 2:
                stage1_checkpoint = torch.load(stage1_path+"/best_checkpoint.pth")
                print('loading level1 iteration' + str(stage1_checkpoint["iter"]))

                if os.path.exists(stage2_path+"/last_checkpoint.pth"):
                    stage2_checkpoint = torch.load(stage2_path+"/last_checkpoint.pth")
                    self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])

                    self.lapnet2_gen.load_state_dict(stage2_checkpoint["generator_state_dict"])
                    self.lapnet2_disc.load_state_dict(stage2_checkpoint["discriminator_state_dict"])
                    self.optimizer_lapnet2_gen.load_state_dict(stage2_checkpoint["gen_optim_state_dict"])
                    self.optimizer_lapnet2_disc.load_state_dict(stage2_checkpoint["disc_optim_state_dict"])

                    self.iter = stage2_checkpoint["iter"]
                    self.best_iter = stage2_checkpoint["best_iter"]
                    self.best_recons_loss = stage2_checkpoint["best_recons_loss"]

                    
                    print('loading level2 iteration' + str(self.iter))
            elif specifics["stage"] == 3:
                stage1_checkpoint = torch.load(stage1_path+"/best_checkpoint.pth")
                stage2_checkpoint = torch.load(stage2_path+"/best_checkpoint.pth")
                

                self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])
                self.lapnet2_gen.load_state_dict(stage2_checkpoint["generator_state_dict"])
                print('loading level1 iteration' + str(stage1_checkpoint["iter"]))
                print('loading level2 iteration' + str(stage2_checkpoint["iter"]))
                if os.path.exists(stage3_path+"/last_checkpoint.pth"):
                    stage3_checkpoint = torch.load(stage3_path+"/last_checkpoint.pth")
                    self.lapnet3_gen.load_state_dict(stage3_checkpoint["generator_state_dict"])
                    self.lapnet3_disc.load_state_dict(stage3_checkpoint["discriminator_state_dict"])
                    self.optimizer_lapnet3_gen.load_state_dict(stage3_checkpoint["gen_optim_state_dict"])
                    self.optimizer_lapnet3_disc.load_state_dict(stage3_checkpoint["disc_optim_state_dict"])

                    self.iter = stage3_checkpoint["iter"]
                    self.best_iter = stage3_checkpoint["best_iter"]
                    self.best_recons_loss = stage3_checkpoint["best_recons_loss"]

                    print('loading level3 iteration' + str(self.iter))
            elif specifics["stage"] == 4:
                stage1_checkpoint = torch.load(stage1_path+"/best_checkpoint.pth")
                stage2_checkpoint = torch.load(stage2_path+"/best_checkpoint.pth")
                stage3_checkpoint = torch.load(stage3_path+"/best_checkpoint.pth")
                

                self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])
                self.lapnet2_gen.load_state_dict(stage2_checkpoint["generator_state_dict"])
                self.lapnet3_gen.load_state_dict(stage3_checkpoint["generator_state_dict"])
                print('loading level1 iteration' + str(stage1_checkpoint["iter"]))
                print('loading level2 iteration' + str(stage2_checkpoint["iter"]))
                print('loading level3 iteration' + str(stage3_checkpoint["iter"]))
                if os.path.exists(stage4_path+"/last_checkpoint.pth"):
                    stage4_checkpoint = torch.load(stage4_path+"/last_checkpoint.pth")
                    self.lapnet4_gen.load_state_dict(stage4_checkpoint["generator_state_dict"])
                    self.lapnet4_disc.load_state_dict(stage4_checkpoint["discriminator_state_dict"])
                    self.optimizer_lapnet4_gen.load_state_dict(stage4_checkpoint["gen_optim_state_dict"])
                    self.optimizer_lapnet4_disc.load_state_dict(stage4_checkpoint["disc_optim_state_dict"])

                    self.iter = stage4_checkpoint["iter"]
                    self.best_iter = stage4_checkpoint["best_iter"]
                    self.best_recons_loss = stage4_checkpoint["best_recons_loss"]

                    print('loading level4 iteration' + str(self.iter))
            elif specifics["stage"] == 5:
                stage1_checkpoint = torch.load(stage1_path+"/best_checkpoint.pth")
                stage2_checkpoint = torch.load(stage2_path+"/best_checkpoint.pth")
                stage3_checkpoint = torch.load(stage3_path+"/best_checkpoint.pth")
                stage4_checkpoint = torch.load(stage4_path+"/best_checkpoint.pth")

                self.lapnet1_gen.load_state_dict(stage1_checkpoint["generator_state_dict"])
                self.lapnet2_gen.load_state_dict(stage2_checkpoint["generator_state_dict"])
                self.lapnet3_gen.load_state_dict(stage3_checkpoint["generator_state_dict"])
                self.lapnet4_gen.load_state_dict(stage4_checkpoint["generator_state_dict"])
                print('loading level1 iteration' + str(stage1_checkpoint["iter"]))
                print('loading level2 iteration' + str(stage2_checkpoint["iter"]))
                print('loading level3 iteration' + str(stage3_checkpoint["iter"]))
                print('loading level4 iteration' + str(stage4_checkpoint["iter"]))
    def run(self, stage, specifics):
        # run the training/testing. print the result.
        if (stage == "training"):
            print("Training...")
            self.train(specifics)
        elif(stage == "testing"):
            print("Testing...")
            errD_fake_mses = []
            PSNRs = []
            SSIMs = []
            reconst_times = []
            with torch.no_grad():
                for idx, (data, _) in enumerate(self.testloader, 0):
                    data_array = data.numpy()
                    if (data_array.shape[0] < self.batch_size):
                        continue
                    for i in range(self.batch_size):
                        target_temp = data_array[i]  # 1x64x64
                        if (self.input_width == 64):
                            self.g4_target[i] = torch.from_numpy(target_temp)  # 3x64x64
                        elif (self.input_width ==32):
                            self.g3_target[i] = torch.from_numpy(target_temp)  # 3x64x64
                    
                    data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                    self.g1_input = self.sensing_matrix1.forward(data)
                    self.y2 =  self.sensing_matrix2.forward(data)
                    self.y3=  self.sensing_matrix3.forward(data)
                    if (self.input_width ==64):
                        self.y4 =  self.sensing_matrix4.forward(data)


                    with torch.no_grad():
                        benchmarkTimeStart = time.time()
                        self.g2_input = self.lapnet1_gen(self.g1_input)
                        self.g3_input = self.lapnet2_gen(self.g2_input, self.y2)
                        self.g4_input = self.lapnet3_gen(self.g3_input, self.y3)
                        if (self.input_width ==64):
                            self.g4_output = self.lapnet4_gen(self.g4_input, self.y4)
                        benchmarkTimeEnd=time.time()

                        if (self.input_width ==32):
                            output = self.g4_input
                            target = self.g3_target
                        if (self.input_width ==64):
                            output =  self.g4_output
                            target=self.g4_target
                        errD_fake_mse = self.criterion_mse(output, target)


                        errD_fake_mses.append(errD_fake_mse)
                        PSNR = utils.compute_average_psnr(target,output)
                        SSIM = utils.compute_average_SSIM(target,output)
                        reconst_time = (benchmarkTimeEnd-benchmarkTimeStart)/self.batch_size
                        errD_fake_mses.append(errD_fake_mse)
                        PSNRs.append(PSNR)
                        SSIMs.append(SSIM)
                        reconst_times.append(reconst_time)
                    if idx % 20 == 0:
                        print('Test: [%d/%d] errG_mse: %.4f \n' % (idx, len(self.testloader), errD_fake_mse.item()))
                        print('Test: [%d/%d] PSNR: %.4f \n' % (idx, len(self.testloader), PSNR))
                        print('Test: [%d/%d] SSIM: %.4f \n' % (idx, len(self.testloader), SSIM))
                        print('Test: [%d/%d] reconst_time: %.4f \n\n' % (idx, len(self.testloader), reconst_time))
        print('Test: average errG_mse: %.4f,' % (sum(errD_fake_mses)/(len(errD_fake_mses))))
        print('Test: average PSNR: %.4f,' % (sum(PSNRs)/len(PSNRs)))
        print('Test: average SSIM: %.4f,' % (sum(SSIMs)/len(SSIMs)))
        print('Test: average reconstruction time: %.12f,' % (sum(reconst_times)/len(reconst_times)))

    def val(self, epoch, stage, channels, valloader, sensing_matrix1, sensing_matrix2, sensing_matrix3, sensing_matrix4,
        target, g1_input, lapnet1_gen, lapnet2_gen, lapnet3_gen, lapnet4_gen, criterion_mse, y2, y3, y4, specifics):
        logOutput = "\n\nVALIDATION"
        with torch.no_grad():
            errD_fake_mses = []
            PSNRs = []
            SSIMs = []
            for idx, (data, _) in enumerate(valloader, 0):
                if data.size(0) != specifics["batch_size"]:
                    continue

                data_array = data.numpy()
                for i in range(specifics["batch_size"]):
                    g4_target_temp = data_array[i]  # 1x64x64

                    if (self.input_width == 64): #r is the base rate that is used to only sample every r-th data entry from the images. To produce the desired sizes for each stage, it must be different for 32x32 and 64x64 images
                            r = 2
                    elif(self.input_width==32):
                        r = 1
                    if stage == 1:
                        target[i] = torch.from_numpy(g4_target_temp[:, ::r*4, ::r*4])  #3x8x8
                    elif stage == 2:
                        target[i] = torch.from_numpy(g4_target_temp[:, ::r*2, ::r*2])  # 3x16x16
                    elif stage == 3:
                        target[i] = torch.from_numpy(g4_target_temp[:, ::r, ::r])  # 3x32x32
                    elif stage == 4:
                        target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64
                    elif stage == 5:
                        target[i] = torch.from_numpy(g4_target_temp)  # 3x64x64
                data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                g1_input = sensing_matrix1.forward(data)
                y2 =  sensing_matrix2.forward(data)
                y3 =  sensing_matrix3.forward(data)
                y4 =  sensing_matrix4.forward(data)
            
                if stage == 1:
                    output = lapnet1_gen(g1_input)
                elif stage == 2:
                    g2_input = lapnet1_gen(g1_input)
                    output = lapnet2_gen(g2_input, y2)
                elif stage == 3:
                    g2_input = lapnet1_gen(g1_input)
                    g3_input = lapnet2_gen(g2_input, y2)
                    output = lapnet3_gen(g3_input, y3)
                elif stage == 4:
                    g2_input = lapnet1_gen(g1_input)
                    g3_input = lapnet2_gen(g2_input, y2)
                    g4_input = lapnet3_gen(g3_input, y3)
                    output = lapnet4_gen(g4_input, y4)

                errD_fake_mse = criterion_mse(output, target)
                PSNR = utils.compute_average_psnr(target,output)
                SSIM = utils.compute_average_SSIM(target,output)
                errD_fake_mses.append(errD_fake_mse)
                PSNRs.append(PSNR)
                SSIMs.append(SSIM)
                if idx % 20 == 0:
                    logOutput += ('\nVal: [%d][%d/%d] errG_mse: %.4f \n' % (epoch, idx, len(valloader), errD_fake_mse.item()))
                    print('Val: [%d][%d/%d] errG_mse: %.4f \n' % (epoch, idx, len(valloader), errD_fake_mse.item()))
            average_recons_loss = sum(errD_fake_mses)/len(errD_fake_mses)
            logOutput = ('\n\nVal: [%d] average errG_mse: %.4f' % (epoch, average_recons_loss))
            logOutput += ('\nVal: [%d] average PSNR: %.4f' % (epoch, sum(PSNRs)/len(PSNRs)))  
            logOutput += ('\nVal: [%d] average SSIM: %.4f' % (epoch, sum(SSIMs)/len(SSIMs)))  
            logOutput += "\nBest average errG_mse so far: " + str(float(self.best_recons_loss)) + " at iteration " + str(self.best_iter)
            logOutput += "\nDifference: " +str(float(average_recons_loss- self.best_recons_loss)) +"\n\n"
            print(logOutput)
            logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], specifics['stage']),"a") 
            logFile.write(logOutput)
            logFile.close()
            vutils.save_image(target.data, '%s/%s/cr%s/%s/stage%s/image/val_epoch_%03d_real.png'
                            % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], specifics["stage"], epoch), normalize=True)
            vutils.save_image(output.data, '%s/%s/cr%s/%s/stage%s/image/val_epoch_%03d_fake.png'
                            % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], specifics["stage"], epoch), normalize=True)
            return average_recons_loss
    def train(self, specifics):
        #input, _ = self.trainloader.__iter__().__next__()
        #input = input.numpy()
        while self.iter < (specifics["epochs"]):
            # training level 1
            if specifics["stage"] == 1 or specifics["stage"] == 5:
                for idx, (data, _) in enumerate(self.trainloader, 0):
                    if data.size(0) != specifics["batch_size"]:
                        continue
                   # print(data.shape)
                    self.lapnet1_gen.train(), self.lapnet1_disc.train()
                    data_array = data.numpy()
                    for i in range(specifics["batch_size"]):
                        g4_target_temp = data_array[i]  # 3x64x64
                        # g3_target_temp = g4_target_temp[:, ::2, ::2]  # 3x32x32
                        # g2_target_temp = g3_target_temp[:, ::2, ::2]  # 3x16x16
                        if (self.input_width == 64): #r is the base rate that is used to only sample every r-th data entry from the images. To produce the desired sizes for each stage, it must be different for 32x32 and 64x64 images
                            r = 2
                        elif(self.input_width==32):
                            r = 1
                        g1_target_temp = g4_target_temp[:, ::r*4, ::r*4]  # 3x8x8
                        self.g1_target[i] = torch.from_numpy(g1_target_temp)
                    with torch.no_grad():
                        data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                        self.g1_input = self.sensing_matrix1.forward(data)
        
                    # Train disc1 with true images
                    self.lapnet1_disc.zero_grad()
                    d1_output = self.lapnet1_disc(self.g1_target)
                    d1_label = self.label.fill_(self.real_label)
                    errD_d1_real_bce = self.criterion_bce(d1_output, d1_label)
                    errD_d1_real_bce.backward()
                    d1_real_mean = d1_output.data.mean()

                    # Train disc1 with fake images
                    g1_output = self.lapnet1_gen(self.g1_input)
                    d1_output = self.lapnet1_disc(g1_output.detach())
                    d1_label = self.label.fill_(self.fake_label)
                    errD_d1_fake_bce = self.criterion_bce(d1_output, d1_label)
                    errD_d1_fake_bce.backward()
                    self.optimizer_lapnet1_disc.step()

                    # Train gen1 with fake images
                    self.lapnet1_gen.zero_grad()
                    d1_label = self.label.fill_(self.real_label)
                    d1_output = self.lapnet1_disc(g1_output)
                    errD_g1_fake_bce = self.criterion_bce(d1_output, d1_label)
                    errD_g1_fake_mse = self.criterion_mse(g1_output, self.g1_target)
                    errD_g1 = specifics["w-loss"] * errD_g1_fake_bce + (1 - specifics["w-loss"]) * errD_g1_fake_mse
                    errD_g1.backward()
                    self.optimizer_lapnet1_gen.step()
                    d1_fake_mean = d1_output.data.mean()

                    if idx % specifics["log-interval"] == 0:
                        logOutput = ('\nLevel %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                            'D(x): %.4f, D(G(z)): %.4f' % (
                                1, self.iter, specifics["epochs"], idx, len(self.trainloader),
                                errD_d1_real_bce.item(),
                                errD_d1_fake_bce.item(),
                                errD_g1_fake_bce.item(),
                                errD_g1_fake_mse.item(),
                                d1_real_mean,
                                d1_fake_mean))
                        print(logOutput)
                        logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1),"a") 
                        logFile.write(logOutput)
                        logFile.close()
                        
                
                torch.save({
                    'iter': self.iter,
                    'best_iter':self.best_iter,
                    'best_recons_loss': self.best_recons_loss,
                    'generator_state_dict': self.lapnet1_gen.state_dict(),
                    'discriminator_state_dict': self.lapnet1_disc.state_dict(),
                    'gen_optim_state_dict':self.optimizer_lapnet1_gen.state_dict(),
                    'disc_optim_state_dict':self.optimizer_lapnet1_disc.state_dict(),
                }, '%s/%s/cr%s/%s/stage%s/model/last_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1))


                vutils.save_image(self.g1_target.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1, self.iter), normalize=True)
                vutils.save_image(g1_output.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1, self.iter), normalize=True)

                self.lapnet1_gen.eval(), self.lapnet1_disc.eval()
                avg_recons_loss = self.val(self.iter, 1, self.channels, self.valloader, self.sensing_matrix1, self.sensing_matrix2, self.sensing_matrix3,
                    self.sensing_matrix4, self.g1_target, self.g1_input, self.lapnet1_gen, self.lapnet2_gen, self.lapnet3_gen, self.lapnet4_gen,
                    self.criterion_mse, self.y2, self.y3, self.y4, specifics)
                if (avg_recons_loss <= self.best_recons_loss):
                    logOutput = "\nSaving current iteration as best iteration...\n\n"
                    print(logOutput)
                    logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1),"a") 
                    logFile.write(logOutput)
                    logFile.close()
                    self.best_iter = self.iter
                    self.best_recons_loss = avg_recons_loss
                    torch.save({
                        'iter': self.iter,
                        'best_iter':self.best_iter,
                        'best_recons_loss': self.best_recons_loss,
                        'generator_state_dict': self.lapnet1_gen.state_dict(),
                        'discriminator_state_dict': self.lapnet1_disc.state_dict(),
                        'gen_optim_state_dict':self.optimizer_lapnet1_gen.state_dict(),
                        'disc_optim_state_dict':self.optimizer_lapnet1_disc.state_dict(),
                    }, '%s/%s/cr%s/%s/stage%s/model/best_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 1))

            # training level 2
            # load weight of level 1
            if specifics["stage"] == 2 or specifics["stage"] == 5:
                for idx, (data, _) in enumerate(self.trainloader, 0):
                    if data.size(0) != specifics["batch_size"]:
                        continue

                    self.lapnet2_gen.train(), self.lapnet2_disc.train()
                    data_array = data.numpy()
                    for i in range(specifics["batch_size"]):
                        if (self.input_width == 64): #r is the base rate that is used to only sample every r-th data entry from the images. To produce the desired sizes for each stage, it must be different for 32x32 and 64x64 images
                            r = 2
                        elif(self.input_width==32):
                            r = 1
                        g4_target_temp = data_array[i]  # 3x64x64
                        #g3_target_temp = g4_target_temp[:, ::r, ::r]  # 3x32x32 
                        g2_target_temp = g4_target_temp[:, ::r*2, ::r*2]  # 3x16x16 
                        g1_target_temp = g4_target_temp[:, ::r*4, ::r*4]  # 3x8x8

                        self.g2_target[i] = torch.from_numpy(g2_target_temp)
                        self.g1_target[i] = torch.from_numpy(g1_target_temp)
                        

                    with torch.no_grad():
                        data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                        self.g1_input = self.sensing_matrix1.forward(data)
                        self.y2 =  self.sensing_matrix2.forward(data)

                    # g2_input = lapnet1_gen(g1_input_var.detach())
                    self.g2_input = self.lapnet1_gen(self.g1_input)
                    # Train disc2 with true images
                    self.lapnet2_disc.zero_grad()
                    d2_output = self.lapnet2_disc(self.g2_target)
                    d2_label = self.label.fill_(self.real_label)
                    errD_d2_real_bce = self.criterion_bce(d2_output, d2_label)
                    errD_d2_real_bce.backward()
                    d2_real_mean = d2_output.data.mean()

                    # Train disc2 with fake images
                    g2_output = self.lapnet2_gen(self.g2_input, self.y2)
                    d2_output = self.lapnet2_disc(g2_output.detach())
                    d2_label = self.label.fill_(self.fake_label)
                    errD_d2_fake_bce = self.criterion_bce(d2_output, d2_label)
                    errD_d2_fake_bce.backward()
                    self.optimizer_lapnet2_disc.step()

                    # Train gen2 with fake images, disc2 is not updated
                    self.lapnet2_gen.zero_grad()
                    d2_label = self.label.fill_(self.real_label)
                    d2_output = self.lapnet2_disc(g2_output)
                    errD_g2_fake_bce = self.criterion_bce(d2_output, d2_label)
                    errD_g2_fake_mse = self.criterion_mse(g2_output, self.g2_target)
                    errD_g2 = specifics["w-loss"] * errD_g2_fake_bce + (1 - specifics["w-loss"]) * errD_g2_fake_mse
                    errD_g2.backward()

                    # optimizer_lapnet1_gen.step()
                    self.optimizer_lapnet2_gen.step()
                    d2_fake_mean = d2_output.data.mean()

                    if idx % specifics["log-interval"] == 0:
                        errD_g1_fake_mse = self.criterion_mse(self.g2_input, self.g1_target)

                        logOutput = ('\nLevel %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                            'errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                                2, self.iter, specifics["epochs"], idx, len(self.trainloader),
                                errD_d2_real_bce.item(),
                                errD_d2_fake_bce.item(),
                                errD_g2_fake_bce.item(),
                                errD_g2_fake_mse.item(),
                                errD_g1_fake_mse.item(),
                                d2_real_mean,
                                d2_fake_mean))
                        print(logOutput)
                        logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2),"a") 
                        logFile.write(logOutput)
                        logFile.close()
                torch.save({
                    'iter': self.iter,
                    'best_iter':self.best_iter,
                    'best_recons_loss': self.best_recons_loss,
                    'generator_state_dict': self.lapnet2_gen.state_dict(),
                    'discriminator_state_dict': self.lapnet2_disc.state_dict(),
                    'gen_optim_state_dict':self.optimizer_lapnet2_gen.state_dict(),
                    'disc_optim_state_dict':self.optimizer_lapnet2_disc.state_dict(),
                }, '%s/%s/cr%s/%s/stage%s/model/last_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2))


                vutils.save_image(self.g2_target.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2, self.iter), normalize=True)
                vutils.save_image(g2_output.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2, self.iter), normalize=True)

                self.lapnet2_gen.eval(), self.lapnet2_disc.eval()
                avg_recons_loss = self.val(self.iter, 2, self.channels, self.valloader, self.sensing_matrix1, self.sensing_matrix2, self.sensing_matrix3,
                    self.sensing_matrix4, self.g2_target, self.g1_input, self.lapnet1_gen, self.lapnet2_gen, self.lapnet3_gen, self.lapnet4_gen,
                    self.criterion_mse, self.y2, self.y3, self.y4, specifics)

                if (avg_recons_loss <= self.best_recons_loss):
                    logOutput = "\nSaving current iteration as best iteration...\n\n"
                    print(logOutput)
                    logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2),"a") 
                    logFile.write(logOutput)
                    logFile.close()
                    self.best_iter = self.iter
                    self.best_recons_loss = avg_recons_loss
                    torch.save({
                        'iter': self.iter,
                        'best_iter':self.best_iter,
                        'best_recons_loss': self.best_recons_loss,
                        'generator_state_dict': self.lapnet2_gen.state_dict(),
                        'discriminator_state_dict': self.lapnet2_disc.state_dict(),
                        'gen_optim_state_dict':self.optimizer_lapnet2_gen.state_dict(),
                        'disc_optim_state_dict':self.optimizer_lapnet2_disc.state_dict(),
                    }, '%s/%s/cr%s/%s/stage%s/model/best_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 2))


            # training level 3
            if specifics["stage"] == 3 or specifics["stage"] == 5:
                for idx, (data, _) in enumerate(self.trainloader, 0):
                    if data.size(0) != specifics["batch_size"]:
                        continue

                    self.lapnet3_gen.train(), self.lapnet3_disc.train()
                    data_array = data.numpy()
                    for i in range (specifics["batch_size"]):

                        if (self.input_width == 64): #r is the base rate that is used to only sample every r-th data entry from the images. To produce the desired sizes for each stage, it must be different for 32x32 and 64x64 images
                            r = 2
                        elif(self.input_width==32):
                            r = 1
                        g4_target_temp = data_array[i]  # 3x64x64
                        g3_target_temp = g4_target_temp[:, ::r, ::r]  # 3x32x32 
                        g2_target_temp = g4_target_temp[:, ::r*2, ::r*2]  # 3x16x16 
                        g1_target_temp = g4_target_temp[:, ::r*4, ::r*4]  # 3x8x8

                        self.g3_target[i] = torch.from_numpy(g3_target_temp)
                        self.g2_target[i] = torch.from_numpy(g2_target_temp)
                        self.g1_target[i] = torch.from_numpy(g1_target_temp)


                    with torch.no_grad():
                        data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                        self.g1_input = self.sensing_matrix1.forward(data)
                        self.y2 =  self.sensing_matrix2.forward(data)
                        self.y3 =  self.sensing_matrix3.forward(data)


                    self.g2_input = self.lapnet1_gen(self.g1_input)  # 1x8x8
                    self.g3_input = self.lapnet2_gen(self.g2_input, self.y2)  # 1x16x16

                    # Train disc3 with true images
                    self.lapnet3_disc.zero_grad()
                    d3_output = self.lapnet3_disc(self.g3_target)
                    d3_label = self.label.fill_(self.real_label)
                    errD_d3_real_bce = self.criterion_bce(d3_output, d3_label)
                    errD_d3_real_bce.backward()
                    d3_real_mean = d3_output.data.mean()
                    # Train disc3 with fake images
                    g3_output = self.lapnet3_gen(self.g3_input, self.y3)
                    d3_output = self.lapnet3_disc(g3_output.detach())
                    d3_label = self.label.fill_(self.fake_label)
                    errD_d3_fake_bce = self.criterion_bce(d3_output, d3_label)
                    errD_d3_fake_bce.backward()
                    self.optimizer_lapnet3_disc.step()
                    # Train gen3 with fake images, disc3 is not updated
                    self.lapnet3_gen.zero_grad()
                    d3_label = self.label.fill_(self.real_label)
                    d3_output = self.lapnet3_disc(g3_output)
                    errD_g3_fake_bce = self.criterion_bce(d3_output, d3_label)
                    errD_g3_fake_mse = self.criterion_mse(g3_output, self.g3_target)
                    errD_g3 = specifics["w-loss"] * errD_g3_fake_bce + (1 - specifics["w-loss"] ) * errD_g3_fake_mse
                    errD_g3.backward()
                    self.optimizer_lapnet3_gen.step()
                    d3_fake_mean = d3_output.data.mean()

                    if idx % specifics["log-interval"] == 0:
                        errD_g1_fake_mse = self.criterion_mse(self.g2_input, self.g1_target)
                        errD_g2_fake_mse = self.criterion_mse(self.g3_input, self.g2_target)

                        logOutput = ('\nLevel %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                            'errG2_mse: %.4f, errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                                3, self.iter, specifics["epochs"], idx, len(self.trainloader),
                                errD_d3_real_bce.item(),
                                errD_d3_fake_bce.item(),
                                errD_g3_fake_bce.item(),
                                errD_g3_fake_mse.item(),
                                errD_g2_fake_mse.item(),
                                errD_g1_fake_mse.item(),
                                d3_real_mean,
                                d3_fake_mean))
                        print(logOutput)
                        logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3),"a") 
                        logFile.write(logOutput)
                        logFile.close()
                torch.save({
                    'iter': self.iter,
                    'best_iter':self.best_iter,
                    'best_recons_loss': self.best_recons_loss,
                    'generator_state_dict': self.lapnet3_gen.state_dict(),
                    'discriminator_state_dict': self.lapnet3_disc.state_dict(),
                    'gen_optim_state_dict':self.optimizer_lapnet3_gen.state_dict(),
                    'disc_optim_state_dict':self.optimizer_lapnet3_disc.state_dict(),
                }, '%s/%s/cr%s/%s/stage%s/model/last_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3))


                vutils.save_image(self.g3_target.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3, self.iter), normalize=True)
                vutils.save_image(g3_output.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3, self.iter), normalize=True)

                self.lapnet3_gen.eval(), self.lapnet3_disc.eval()
                avg_recons_loss= self.val(self.iter, 3, self.channels, self.valloader, self.sensing_matrix1, self.sensing_matrix2, self.sensing_matrix3,
                    self.sensing_matrix4, self.g3_target, self.g1_input, self.lapnet1_gen, self.lapnet2_gen, self.lapnet3_gen, self.lapnet4_gen,
                    self.criterion_mse, self.y2, self.y3, self.y4, specifics)
                if (avg_recons_loss <= self.best_recons_loss):
                    logOutput = "\nSaving current iteration as best iteration...\n\n"
                    print(logOutput)
                    logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3),"a") 
                    logFile.write(logOutput)
                    logFile.close()
                    self.best_iter = self.iter
                    self.best_recons_loss = avg_recons_loss
                    torch.save({
                        'iter': self.iter,
                        'best_iter':self.best_iter,
                        'best_recons_loss': self.best_recons_loss,
                        'generator_state_dict': self.lapnet3_gen.state_dict(),
                        'discriminator_state_dict': self.lapnet3_disc.state_dict(),
                        'gen_optim_state_dict':self.optimizer_lapnet3_gen.state_dict(),
                        'disc_optim_state_dict':self.optimizer_lapnet3_disc.state_dict(),
                    }, '%s/%s/cr%s/%s/stage%s/model/best_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 3))

            # training level 4
            if specifics["stage"] == 4 or specifics["stage"] == 5:
                for idx, (data, _) in enumerate(self.trainloader, 0):
                    if data.size(0) != specifics["batch_size"]:
                        continue

                    self.lapnet4_gen.train(), self.lapnet4_disc.train()
                    data_array = data.numpy()
                    for i in range(specifics["batch_size"]):
                        g4_target_temp = data_array[i]  # 1x64x64
                        g3_target_temp = g4_target_temp[:, ::2, ::2]  # 1x32x32
                        g2_target_temp = g3_target_temp[:, ::2, ::2]  # 1x16x16
                        g1_target_temp = g2_target_temp[:, ::2, ::2]  # 1x8x8

                        self.g4_target[i] = torch.from_numpy(g4_target_temp)
                        self.g3_target[i] = torch.from_numpy(g3_target_temp)
                        self.g2_target[i] = torch.from_numpy(g2_target_temp)
                        self.g1_target[i] = torch.from_numpy(g1_target_temp)

                    with torch.no_grad():
                        data = torch.from_numpy(data_array.reshape(self.batch_size,self.channels,-1)).cuda()
                        self.g1_input = self.sensing_matrix1.forward(data)
                        self.y2 =  self.sensing_matrix2.forward(data)
                        self.y3 =  self.sensing_matrix3.forward(data)
                        self.y4 =  self.sensing_matrix4.forward(data)

                    self.g2_input = self.lapnet1_gen(self.g1_input)  # 1x8x8
                    self.g3_input = self.lapnet2_gen(self.g2_input, self.y2)  # 1x16x16
                    self.g4_input = self.lapnet3_gen(self.g3_input, self.y3)  # 1x32x32

                    # Train disc4 with true images
                    self.lapnet4_disc.zero_grad()
                    d4_output = self.lapnet4_disc(self.g4_target)
                    d4_label = self.label.fill_(self.real_label)
                    errD_d4_real_bce = self.criterion_bce(d4_output, d4_label)
                    errD_d4_real_bce.backward()
                    d4_real_mean = d4_output.data.mean()
                    # Train disc4 with fake images
                    g4_output = self.lapnet4_gen(self.g4_input, self.y4)
                    d4_output = self.lapnet4_disc(g4_output.detach())
                    d4_label = self.label.fill_(self.fake_label)
                    errD_d4_fake_bce = self.criterion_bce(d4_output, d4_label)
                    errD_d4_fake_bce.backward()
                    self.optimizer_lapnet4_disc.step()
                    # Train gen4 with fake images, disc4 is not updated
                    self.lapnet4_gen.zero_grad()
                    d4_label = self.label.fill_(self.real_label)
                    d4_output = self.lapnet4_disc(g4_output)
                    errD_g4_fake_bce = self.criterion_bce(d4_output, d4_label)
                    errD_g4_fake_mse = self.criterion_mse(g4_output, self.g4_target)
                    errD_g4 = specifics["w-loss"] * errD_g4_fake_bce + (1 - specifics["w-loss"]) * errD_g4_fake_mse
                    errD_g4.backward()
                    self.optimizer_lapnet4_gen.step()
                    d4_fake_mean = d4_output.data.mean()

                    if idx % specifics["log-interval"] == 0:
                        errD_g1_fake_mse = self.criterion_mse(self.g2_input, self.g1_target)
                        errD_g2_fake_mse = self.criterion_mse(self.g3_input, self.g2_target)
                        errD_g3_fake_mse = self.criterion_mse(self.g4_input, self.g3_target)

                        logOutput = ('\nLevel %d [%d/%d][%d/%d] errD_real: %.4f, errD_fake: %.4f, errG_bce: %.4f errG_mse: %.4f,'
                            'errG3_mse: %.4f, errG2_mse: %.4f, errG1_mse: %.4f, D(x): %.4f, D(G(z)): %.4f' % (
                                4, self.iter, specifics["epochs"], idx, len(self.trainloader),
                                errD_d4_real_bce.item(),
                                errD_d4_fake_bce.item(),
                                errD_g4_fake_bce.item(),
                                errD_g4_fake_mse.item(),
                                errD_g3_fake_mse.item(),
                                errD_g2_fake_mse.item(),
                                errD_g1_fake_mse.item(),
                                d4_real_mean,
                                d4_fake_mean))
                        print(logOutput)
                        logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4),"a") 
                        logFile.write(logOutput)
                        logFile.close()

                torch.save({
                    'iter': self.iter,
                    'best_iter':self.best_iter,
                    'best_recons_loss': self.best_recons_loss,
                    'generator_state_dict': self.lapnet4_gen.state_dict(),
                    'discriminator_state_dict': self.lapnet4_disc.state_dict(),
                    'gen_optim_state_dict':self.optimizer_lapnet4_gen.state_dict(),
                    'disc_optim_state_dict':self.optimizer_lapnet4_disc.state_dict(),
                }, '%s/%s/cr%s/%s/stage%s/model/last_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4))

                vutils.save_image(self.g4_target.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_real.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4, self.iter), normalize=True)
                vutils.save_image(g4_output.data,
                                '%s/%s/cr%s/%s/stage%s/image/epoch_%03d_fake.png'
                                % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4, self.iter), normalize=True)

                self.lapnet4_gen.eval(), self.lapnet4_disc.eval()
                avg_recons_loss = self.val(self.iter, 4, self.channels, self.valloader, self.sensing_matrix1, self.sensing_matrix2, self.sensing_matrix3,
                    self.sensing_matrix4, self.g4_target, self.g1_input, self.lapnet1_gen, self.lapnet2_gen, self.lapnet3_gen, self.lapnet4_gen,
                    self.criterion_mse, self.y2, self.y3, self.y4, specifics)
                if (avg_recons_loss <= self.best_recons_loss):
                    logOutput = "\nSaving current iteration as best iteration...\n\n"
                    print(logOutput)
                    logFile = open('%s/%s/cr%s/%s/stage%s/model/log.txt' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4),"a") 
                    logFile.write(logOutput)
                    logFile.close()
                    self.best_iter = self.iter
                    self.best_recons_loss = avg_recons_loss
                    torch.save({
                        'iter': self.iter,
                        'best_iter':self.best_iter,
                        'best_recons_loss': self.best_recons_loss,
                        'generator_state_dict': self.lapnet4_gen.state_dict(),
                        'discriminator_state_dict': self.lapnet4_disc.state_dict(),
                        'gen_optim_state_dict':self.optimizer_lapnet4_gen.state_dict(),
                        'disc_optim_state_dict':self.optimizer_lapnet4_disc.state_dict(),
                    }, '%s/%s/cr%s/%s/stage%s/model/best_checkpoint.pth' % (specifics["outf"], specifics["dataset"], self.cr4, specifics["model"], 4))

            self.iter+=1
    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.uniform(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            init.uniform(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform(m.weight.data, 0.0, 0.02)
            init.constant(m.bias.data, 0.0)
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)


    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm2d') != -1:
            init.uniform(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)


    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv') != -1:
            init.orthogonal(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
            init.orthogonal(m.weight.data, gain=1)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)


    def weights_init(self, net, init_type = 'normal'):
        if init_type == 'normal':
            net.apply(self.weights_init_normal)
        elif init_type == 'xavier':
            net.apply(self.weights_init_xavier)
        elif init_type == 'kaiming':
            net.apply(self.weights_init_kaiming)
        elif init_type == 'orthogonal':
            net.apply(self.weights_init_orthogonal)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
