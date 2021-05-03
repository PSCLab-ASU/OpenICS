import torch
import torchvision
import math
import numpy
from torch import nn
import torch.optim as optim
import CSGAN.utils as utils
from PIL import Image
from numpy.random import randn
import os
import time
def reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing, specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if (reconstruction == 'csgan' or reconstruction == 'CSGAN'):
        CSGAN = csgan(input_channel, input_width, input_height, m, n,sensing, specifics)
        return CSGAN

class csgan():
    def __init__(self, input_channel, input_width, input_height, m, n, sensing, specifics):
        self.channels = input_channel
        self.input_width = input_width
        self.input_height = input_height
        self.m = m 
        self.n = n
        self.batch_size = specifics["batch_size"]
        self.learnedSensing = specifics["learned_sensing_parameters"]
        self.generatorType = specifics["generator"]
        self.lr = specifics['lr']
        self.num_latents = specifics['num_latents']
        self.num_training_iterations = specifics['num_training_iterations']
        self.num_z_iters = specifics['num_z_iters']
        self.z_project_method = specifics['z_project_method']
        self.summary_every_step = specifics['summary_every_step']
        self.output_file = specifics["output_file"]
        self.output_file_best = specifics["output_file_best"]
        self.save_every_step = specifics["save_every_step"]
        self.sensing = sensing
        self.specifics=specifics
        self.val_every_step = specifics["val_every_step"]

    def initialize(self,dset,sensing_method,stage,specifics):
      
        torch.set_default_dtype(torch.float32)
        self.dataset = dset
        #if (self.sensing == "sensing_matrix"):
          #sensing_matrix_np = 3*numpy.ones((self.channels,self.m,self.n))#0.05*randn(self.input_channels, 2, 9) #multiplied by 0.05 means it is a normal distribution with std=0.05
         # with torch.no_grad():
         #     self.sensing_matrix = sensing_method(self.n,self.m,self.input_width,self.channels).cuda()
        #if (self.sensing =="NN_mnist" or self.sensing =="NN_celeba"): #if sensing is NN, initialize the weights appropriatelly 
         # self.discriminator = sensing_method(self.n, self.m, self.input_width, self.channels).cuda()
         # self.discriminator.apply(init_weights)
        #self.sensing_matrix = torch.normal(mean = 0.0, std = 0.05, size = [self.m,self.n]).cuda() #torch.normal(mean = 0.0, std = 0.05, size = [2,9]).cuda()

        if (self.generatorType == "MLP"):
          self.generator = MLPGeneratorNet(self.num_latents,self.n,self.channels,self.input_width).cuda()
        elif(self.generatorType == "DCGAN"):
          if (self.input_width == 64):
            self.generator = csgm_dcgan_gen(self.num_latents,self.n,self.channels).cuda()
          elif(self.input_width == 32):
            self.generator = csgm_dcgan_gen32x32(self.num_latents,self.n,self.channels).cuda()
          else:
            print("DCGAN only supports input images with dimensions 32x32 and 64x64")
            exit()
        self.generator.apply(init_weights)

        self.sensing_method = sensing_method(self.n,self.m,self.input_width,self.channels).cuda()
        if (self.sensing =="NN_MLP" or self.sensing =="NN_DCGAN"): #if sensing is NN, initialize the weights appropriatelly 
         # self.discriminator = sensing_method(self.n, self.m, self.input_width, self.channels).cuda()
          self.sensing_method.apply(init_weights)

        self._log_step_size_module = log_step_size_module(specifics['z_step_size']).cuda()
        self.z_step_size = lambda : torch.exp(self._log_step_size_module.forward(torch.tensor([1.0]).cuda()))
        self.s = None
        
        #if not os.path.exists(self.output_file):
         #   os.makedirs(self.output_file)
        if (stage == "training"):
          self.trainloader = torch.utils.data.DataLoader(self.dataset[0], batch_size=self.batch_size, shuffle=True) 
          self.valloader = torch.utils.data.DataLoader(self.dataset[1], batch_size=self.batch_size, shuffle=False) 
        elif (stage =="testing"):
          self.testloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def measure(self, input):
      #if self.sensing == "sensing_matrix":
         #input = torch.reshape(input, [self.batch_size,self.n,1])
         #output = torch.matmul(self.sensing_matrix, input)
         #output = torch.reshape(output, [self.batch_size,self.m])
      output = self.sensing_method.forward(input)
      return output
    def get_measurement_error(self,target_img,sample_img):
      self.m_targets = self.measure(target_img)
      self.m_samples = self.measure(sample_img)

      return torch.sum(torch.square(self.m_targets-self.m_samples),dim=-1)
    def get_rip_loss(self, img1,img2):
      m1 = self.measure(img1)
      m2 = self.measure(img2)

      img_diff_norm = torch.norm(torch.flatten(img1,start_dim=1)-torch.flatten(img2,start_dim=1),dim=-1,p = None) 
      m_diff_norm = torch.norm(m1-m2,dim =-1,p = None)
      return torch.square(img_diff_norm-m_diff_norm)
    def optimise_and_sample(self,init_z, data, is_training):
      if self.num_z_iters ==0:
        z_final = init_z
        self.s = self.z_step_size()
        samples = self.generator.forward(z_final) 

        return samples, z_final
      else:

        z = _project_z(init_z, self.z_project_method)

        for i in range(self.num_z_iters):

          loop_samples = self.generator.forward(z)

          self.s = self.z_step_size()
          gen_loss = self.get_measurement_error(data, loop_samples)
          

          gen_loss.retain_grad = True
          z_grad = torch.autograd.grad(gen_loss,z,grad_outputs=torch.ones_like(gen_loss), create_graph=True)[0]

          z=z.add(-1*self.s*z_grad)

          z = _project_z(z, self.z_project_method)
         
      z_final = z
      samples = self.generator.forward(z_final)  
      #if not is_training:
       # self.optimizer.zero_grad()
      return samples, z_final
    def run(self, stage):
      if (stage == 'training'):
        trainStartTime = time.time()
        if (self.learnedSensing):
          metric_params = self.sensing_method.parameters()
        else:
          metric_params = []
        params =list(list(self.generator.parameters())+list(self._log_step_size_module.parameters())+list(metric_params))#list(self.discriminator.parameters())+list(self.generator.parameters())+list(self._log_step_size_module.parameters())
        self.optimizer = optim.Adam(params, self.lr, (0.9,0.999), eps = 1e-08)

        prior = utils.make_prior(self.num_latents) 

        iter = 0
        best_iter = 0 #iteration for which the model performed the best (on the validation set)
        best_recons_loss = math.inf #the recons_loss that was measured for the best iteration of the model

        if os.path.exists(self.output_file):
          checkpoint = torch.load(self.output_file)
          self.generator.load_state_dict(checkpoint['generator_state_dict'])
          self.sensing_method.load_state_dict(checkpoint['sensing_method_state_dict'])
          self._log_step_size_module.load_state_dict(checkpoint['_log_step_size_module_state_dict'])
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      
          iter = checkpoint['iter']
          best_iter = checkpoint['best_iter']
          best_recons_loss = checkpoint['best_recons_loss']
          self.generator.train()
          self._log_step_size_module.train()
          if (self.learnedSensing):
            self.sensing_method.train()

        while (iter < self.num_training_iterations):         
          self.optimizer.zero_grad() 
          images, _ = self.trainloader.__iter__().__next__()
          images = images.cuda()
          
          benchmarkTimeStart = time.time()
          generator_inputs = prior.sample([self.batch_size]).cuda()
          generator_inputs.requires_grad = True
          samples, optimised_z = self.optimise_and_sample(generator_inputs,images,True)   
          benchmarkTimeEnd = time.time()
          
          optimisation_cost = torch.mean(torch.sum((optimised_z-generator_inputs)**2,-1))
          initial_samples = self.generator.forward(generator_inputs)
          generator_loss = torch.mean(self.get_measurement_error(images, samples))

          r1 = self.get_rip_loss(samples,initial_samples)
          r2 = self.get_rip_loss(samples,images)
          r3 = self.get_rip_loss(initial_samples, images)
          rip_loss = torch.mean((r1+r2+r3)/3.0)
          total_loss = generator_loss+rip_loss

          total_loss.backward()
          
          self.optimizer.step()

          recons_loss = torch.mean(torch.norm(torch.flatten(samples,start_dim=1)-torch.flatten(images,start_dim=1),dim=-1,p = None))
          if (iter % self.summary_every_step ==0):    
              PSNR = utils.compute_average_psnr(images,samples)
              SSIM = utils.compute_average_SSIM(images,samples)
              logOutput = "\n\n\niteration:" +str(iter)
              logOutput += "\nNumber of measurements: " + str(self.m)
              logOutput += "\n" + "z_step_size: " + str(self.s.item())
              logOutput += "\n" + "reconstruction time: " +str((benchmarkTimeEnd-benchmarkTimeStart)/self.batch_size)
              logOutput += "\n" + "Time since training started:" + (str(time.time()-trainStartTime))
              logOutput += "\n" + "recons_loss: "+ str(recons_loss.item())   
              logOutput += "\n" + "PSNR:" +str(PSNR)
              logOutput += "\n" + "SSIM:" +str(SSIM)
              #print("\n\n\niteration: ", iter)
              #print("z_step_size: ", self.s.item())
             # print("Images mean: ", torch.mean(images).item())
              #print("Initial samples mean: ", torch.mean(initial_samples).item())
              #print("Samples mean: ", torch.mean(samples).item())
              #print("Prior mean: ", torch.mean(generator_inputs).item())
              #print("Optimised z mean: ", torch.mean(optimised_z).item())
              #print("m_targets mean: ", torch.mean(self.m_targets).item())
              #print("m_samples mean: ", torch.mean(self.m_samples).item())
              #print("opt_cost: ", optimisation_cost.item())
              #print("gen loss:", generator_loss.item())
              #print("r1 mean: ", torch.mean(r1).item())
              #print("r2 mean: ", torch.mean(r2).item())
              #print("r3 mean: ", torch.mean(r3).item())
              #print("rip loss: ", rip_loss.item())
              
              #print("recons_loss: ", recons_loss.item())

              print(logOutput)
              logFile = open(self.specifics["log_file"],"a") 
              logFile.write(logOutput)
              logFile.close()

              torchvision.utils.save_image(utils.postprocess(images), "images "+".png")
              torchvision.utils.save_image(utils.postprocess(samples), "samples "+".png")
          if (iter % self.val_every_step == 0):
            
            #self.generator.eval()
            #self.sensing_method.eval()
            

      
            recons_losses = []
            PSNRs = []
            SSIMs = []
            valiter = 0
            for _valimages,_ in self.valloader:
              if _valimages.shape[0] <self.batch_size:
                continue
              valimages = _valimages.cuda()
              valiter +=1
              generator_inputs = prior.sample([self.batch_size]).cuda()
              generator_inputs.requires_grad = True
              samples, optimised_z = self.optimise_and_sample(generator_inputs,valimages,False)   
              with torch.no_grad():  
                
                optimisation_cost = torch.mean(torch.sum((optimised_z-generator_inputs)**2,-1))
                initial_samples = self.generator.forward(generator_inputs)
                generator_loss = torch.mean(self.get_measurement_error(valimages, samples))

                r1 = self.get_rip_loss(samples,initial_samples)
                r2 = self.get_rip_loss(samples,valimages)
                r3 = self.get_rip_loss(initial_samples, valimages)
                rip_loss = torch.mean((r1+r2+r3)/3.0)
                total_loss = generator_loss+rip_loss

                recons_loss = torch.mean(torch.norm(torch.flatten(samples,start_dim=1)-torch.flatten(valimages,start_dim=1),dim=-1,p = None))
                PSNR = utils.compute_average_psnr(valimages,samples)
                SSIM = utils.compute_average_SSIM(valimages,samples)
                logOutput = "\nVALIDATION: "+ str(valiter)+ "/"+str(len(self.valloader)) +"   recons_loss:"+ str(recons_loss.item())
                logOutput += "\nVALIDATION: "+ str(valiter)+ "/"+str(len(self.valloader)) +"   PSNR:"+ str(PSNR)
                logOutput += "\nVALIDATION: "+ str(valiter)+ "/"+str(len(self.valloader)) +"   SSIM:"+ str(SSIM)
                print(logOutput)
                logFile = open(self.specifics["log_file"],"a") 
                logFile.write(logOutput)
                logFile.close()
                recons_losses.append(recons_loss)
                PSNRs.append(PSNR)
                SSIMs.append(SSIM)
              self.generator.zero_grad()
              self.sensing_method.zero_grad()
              #print("z_step_size: ", self.s.item())
             # print("Images mean: ", torch.mean(images).item())
              #print("Initial samples mean: ", torch.mean(initial_samples).item())
              #print("Samples mean: ", torch.mean(samples).item())
              #print("Prior mean: ", torch.mean(generator_inputs).item())
              #print("Optimised z mean: ", torch.mean(optimised_z).item())
              #print("m_targets mean: ", torch.mean(self.m_targets).item())
              #print("m_samples mean: ", torch.mean(self.m_samples).item())
              #print("opt_cost: ", optimisation_cost.item())
              #print("gen loss:", generator_loss.item())
              #print("r1 mean: ", torch.mean(r1).item())
              #print("r2 mean: ", torch.mean(r2).item())
              #print("r3 mean: ", torch.mean(r3).item())
              #print("rip loss: ", rip_loss.item())


              print(logOutput)
              logFile = open(self.specifics["log_file"],"a") 
              logFile.write(logOutput)
              logFile.close()
              
             # print("recons_loss: ", recons_loss.item())
            
            avg_recons_loss = (sum(recons_losses)/len(recons_losses)).item()
            logOutput = "\nRESULTS FOR ITERATION " + str(iter)
            logOutput += "\nVALIDATION AVERAGE RECONSTRUCTION LOSS: "+str(avg_recons_loss)
            logOutput += "\nBEST AVERAGE SO FAR: " +str(best_recons_loss) + " from iteration " + str(best_iter)
            logOutput += "\nDIFFERENCE: " +str(avg_recons_loss- best_recons_loss)
            logOutput += "\nVALIDATION AVERAGE PSNR: "+ str(sum(PSNRs)/len(PSNRs))
            logOutput += "\nVALIDATION AVERAGE SSIM: "+ str(sum(SSIMs)/len(SSIMs))

            if (avg_recons_loss <= best_recons_loss):
              logOutput += "\nSaving current iteration as best iteration..."
              best_iter = iter
              best_recons_loss = avg_recons_loss
              torch.save({
              'iter': iter,
              'best_iter':best_iter,
              'best_recons_loss': best_recons_loss,
              'loss': total_loss,
              'generator_state_dict': self.generator.state_dict(),
              'sensing_method_state_dict': self.sensing_method.state_dict(),
              '_log_step_size_module_state_dict': self._log_step_size_module.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              }, self.output_file_best)
            

            print(logOutput)
            logFile = open(self.specifics["log_file"],"a") 
            logFile.write(logOutput)
            logFile.close()
            torchvision.utils.save_image(utils.postprocess(valimages), "valimages "+".png")
            torchvision.utils.save_image(utils.postprocess(samples), "valsamples "+".png")
            #self.generator.train()
            #self.sensing_method.train()
          if (iter % self.save_every_step ==0):
            torch.save({
              'iter': iter,
              'best_iter':best_iter,
              'best_recons_loss': best_recons_loss,
              'loss': total_loss,
              'generator_state_dict': self.generator.state_dict(),
              'sensing_method_state_dict': self.sensing_method.state_dict(),
              '_log_step_size_module_state_dict': self._log_step_size_module.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.output_file)
          iter+=1
      elif (stage == 'testing'): 
        iter = -1
        if os.path.exists(self.output_file):
          checkpoint = torch.load(self.output_file)
          self.generator.load_state_dict(checkpoint['generator_state_dict'])
          self.sensing_method.load_state_dict(checkpoint['sensing_method_state_dict'])
          self._log_step_size_module.load_state_dict(checkpoint['_log_step_size_module_state_dict'])
  
          #self.generator.eval()
          #self.sensing_method.eval()
          prior = utils.make_prior(self.num_latents) 
          recons_losses = []
          PSNRs = []
          SSIMs = []
          reconst_times = []
          for images,_ in self.testloader:
            if images.shape[0] <self.batch_size:
              continue
            iter+=1
            #images, _ = self.testloader.__iter__().__next__() 
            images = images.cuda()

            benchmarkTimeStart = time.time()

            generator_inputs = prior.sample([self.batch_size]).cuda()
            generator_inputs.requires_grad = True
            samples, optimised_z = self.optimise_and_sample(generator_inputs,images,False)   
            benchmarkTimeEnd = time.time()
            with torch.no_grad():      
              optimisation_cost = torch.mean(torch.sum((optimised_z-generator_inputs)**2,-1))
              initial_samples = self.generator.forward(generator_inputs)
              generator_loss = torch.mean(self.get_measurement_error(images, samples))

              r1 = self.get_rip_loss(samples,initial_samples)
              r2 = self.get_rip_loss(samples,images)
              r3 = self.get_rip_loss(initial_samples, images)
              rip_loss = torch.mean((r1+r2+r3)/3.0)
              total_loss = generator_loss+rip_loss

              recons_loss = torch.mean(torch.norm(torch.flatten(samples,start_dim=1)-torch.flatten(images,start_dim=1),dim=-1,p = None))
              PSNR = utils.compute_average_psnr(images,samples)
              SSIM = utils.compute_average_SSIM(images,samples)
              reconst_time = benchmarkTimeEnd-benchmarkTimeStart
              recons_losses.append(recons_loss)
              PSNRs.append(PSNR)
              SSIMs.append(SSIM)
              reconst_times.append(reconst_time)
              self.generator.zero_grad()
              self.sensing_method.zero_grad()
              print("\n\ntest iteration: ", iter)
            #  print("z_step_size: ", self.s.item())
              
             # print("Images mean: ", torch.mean(images).item())
              #print("Initial samples mean: ", torch.mean(initial_samples).item())
              #print("Samples mean: ", torch.mean(samples).item())
              #print("Prior mean: ", torch.mean(generator_inputs).item())
              #print("Optimised z mean: ", torch.mean(optimised_z).item())
              #print("m_targets mean: ", torch.mean(self.m_targets).item())
              #print("m_samples mean: ", torch.mean(self.m_samples).item())

              #print("opt_cost: ", optimisation_cost.item())
              #print("gen loss:", generator_loss.item())

              #print("r1 mean: ", torch.mean(r1).item())
              #print("r2 mean: ", torch.mean(r2).item())
              #print("r3 mean: ", torch.mean(r3).item())
              
              #print("rip loss: ", rip_loss.item())
              
              print("recons_loss: ", recons_loss.item())
              print("PSNR: ", str(PSNR))
              print("SSIM: "+ str(SSIM))
              print("reconstruction time per image: " +str((reconst_time)/self.batch_size))
              torchvision.utils.save_image(utils.postprocess(images), "testimages "+".png")
              torchvision.utils.save_image(utils.postprocess(samples), "testsamples "+".png")
          print("AVERAGE RECONSTRUCTION LOSS: ", (sum(recons_losses)/len(recons_losses)).item())
          print("AVERAGE PSNR: ", sum(PSNRs)/len(PSNRs))
          print("AVERAGE SSIM: ", sum(SSIMs)/len(SSIMs))
          print("AVERAGE RECONSTRUCTION TIME PER IMAGE: " +str((sum(reconst_times)/(len(reconst_times)*self.batch_size))))
        else:
          print("Error: "+ self.output_file+ " was not found")
      



class log_step_size_module(nn.Module):
  def __init__(self, z_step_size, name='log_step_size_module'):
    super(log_step_size_module, self).__init__()
    self.linear = nn.Linear(1,1,False)
    torch.nn.init.constant_(self.linear.weight,math.log(z_step_size))

  def forward(self, inputs):
    return self.linear(inputs)



#Generator for MNIST dataset
class MLPGeneratorNet(nn.Module):
  """MNIST generator net."""
  def __init__(self, num_inputs, n, channels,imgdim, name='mlp_generator'):
    super(MLPGeneratorNet, self).__init__() 
    self.imgdim = imgdim
    self.channels = channels
    
    self.main = nn.Sequential(
        nn.Linear(num_inputs,500,True),
        nn.LeakyReLU(0.2,inplace=False),
        nn.Linear(500,500,True),
        nn.LeakyReLU(0.2,inplace=False),
        nn.Linear(500,n,True),
        nn.Tanh()
    )

  def forward(self, inputs):
    batchsize = inputs.shape[0]
    out = self.main(inputs)
    return torch.reshape(out, [batchsize,self.channels,self.imgdim,self.imgdim])

class DCGAN(nn.Module):
  """MNIST generator net."""

  def __init__(self, num_inputs, num_outputs, name='dcgan'):
    super(DCGAN, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.lin = nn.Linear(num_inputs,8192,True)
    self.main = nn.Sequential(
        nn.ConvTranspose2d(num_inputs,512,4,1,0,bias = False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512,256,4,2,1,bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256,128,4,2,1,bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128,64,4,2,1,bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64,3,4,2,1,bias = False),
        nn.Tanh()
    )

  def forward(self, inputs):
    out = self.main(inputs)
    return out
class csgm_dcgan_gen(nn.Module): #DCGAN used in the CSGM paper
  """MNIST generator net."""
  def __init__(self, num_inputs, num_outputs,channels, name='sngennet'):
    super(csgm_dcgan_gen, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.lin = nn.Linear(num_inputs,8192,True)
    self.main = nn.Sequential(
        nn.ConvTranspose2d(512,256,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(256,1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(256,128,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(128,1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(128,64,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(64,0.1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(64,channels,5,2,padding=2,output_padding=1),
        nn.Tanh()
    )

  def forward(self, inputs):
    batch_size = inputs.shape[0]
    up_tensor = self.lin(inputs)
    first_tensor = torch.reshape(up_tensor,(batch_size, 512, 4,4))
    out = self.main(first_tensor)
    return out
class csgm_dcgan_gen32x32(nn.Module): #DCGAN used in the CSGM paper
  """MNIST generator net."""
  def __init__(self, num_inputs, num_outputs,channels, name='sngennet'):
    super(csgm_dcgan_gen32x32, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.lin = nn.Linear(num_inputs,2048,True)
    self.main = nn.Sequential(
        nn.ConvTranspose2d(512,256,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(256,1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(256,128,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(128,1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(128,64,5,2,padding=2,output_padding=1),
        nn.BatchNorm2d(64,0.1e-5,0.9,True),
        nn.ReLU(),
        nn.ConvTranspose2d(64,channels,5,2,padding=2,output_padding=1),
        nn.Tanh()
    )

  def forward(self, inputs):
    batch_size = inputs.shape[0]
    up_tensor = self.lin(inputs)
    first_tensor = torch.reshape(up_tensor,(batch_size, 512, 2,2))
    out = self.main(first_tensor)
    return out
class SNGenNet(nn.Module):
  """MNIST generator net."""

  def __init__(self, num_inputs, num_outputs, name='sngennet'):
    super(SNGenNet, self).__init__() #super(MLPGeneratorNet, self).__init__(name=name)
    self.lin = nn.Linear(num_inputs,8192,True)
    self.main = nn.Sequential(
        nn.ConvTranspose2d(512,256,(4,4),2,padding=1),
        nn.BatchNorm2d(256,0.001,0.999,True),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(256,128,(4,4),2, padding = 1),
        nn.BatchNorm2d(128,0.001,0.999,True),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128,64,(4,4),2, padding = 1),
        nn.BatchNorm2d(64,0.001,0.999,True),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64,3,(4,4),2, padding = 1),
        nn.Tanh()
    )

  def forward(self, inputs):
    batch_size = inputs.shape[0]
    up_tensor = self.lin(inputs)
    first_tensor = torch.reshape(up_tensor,(batch_size, 512, 4,4))
    out = self.main(first_tensor)
    #print("OOO",out.shape)
    return out


def init_weights(m):
  if type(m) == nn.ConvTranspose2d:
    #std = 1/math.sqrt(m.in_features)
    #print("WEIGHTT:", std)
    #std = 1/math.sqrt(m.in_features)
    torch.nn.init.zeros_(m.bias)
    torch.nn.init.normal_(m.weight,mean=0.0, std= 0.02)

   # torch.nn.init.constant_(m.weight,0.01)
  if type(m) == nn.Conv2d:

    torch.nn.init.zeros_(m.bias)
    torch.nn.init.normal_(m.weight,mean=0.0, std= 0.02)

  if type(m) == nn.Linear:
    #std = 1/math.sqrt(m.in_features)

    torch.nn.init.zeros_(m.bias)
    torch.nn.init.normal_(m.weight,mean=0.0, std= 0.02)



def _project_z(z, project_method='clip'):
  """To be used for projected gradient descent over z."""
  if project_method == 'norm':
    z_p = torch.nn.functional.normalize(z, p=2, dim=-1)
  #elif project_method == 'clip': #not reimplemented yet
   # z_p = tf.clip_by_value(z, -1, 1)
  else:
    raise ValueError('Unknown project_method: {}'.format(project_method))
  return z_p

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
  #Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
  def norm_cdf(x):
      #Computes standard normal cumulative distribution function
      return (1+math.erf(x/math.sqrt(2.)))/2.

  with torch.no_grad():
      l = norm_cdf((a-mean)/std)
      u = norm_cdf((b-mean)/std)
      sqrt_two = math.sqrt(2.)

      #First, fill with uniform values
      tensor.uniform_(0., 1.)

      #Scale by 2(l-u), shift by 2l-1
      tensor.mul_(2*(u-l))
      tensor.add_(2*l-1)

      #Ensure that the values are strictly between -1 and 1
      eps = torch.finfo(tensor.dtype).eps
      tensor.clamp_(min=-(1.-eps), max=(1.-eps))

      #Now use the inverse erf to get distributed values
      tensor.erfinv_()

      #Clamp one last time to ensure it's still in the proper range
      tensor.clamp_(min=a, max=b)
      return tensor
