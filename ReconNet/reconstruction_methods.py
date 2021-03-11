import torch
import torchvision

import os
import time
import numpy as np

import ReconNet.utils as u

model_ext = '.rnet'
best_model_name = 'best_model' + model_ext
sensing_ext = '.sensing'
sensing_name = 'rand.rnet' + sensing_ext

def reconstruction_method(reconstruction,input_channel,input_width,input_height,m,n,specifics):
    if reconstruction.lower() == 'reconnet':
        return ReconNetWrapper(input_channel,input_width,input_height,m,n,specifics)

class ReconNetWrapper():
    def __init__(self,input_channel,input_width,input_height,m,n,specifics):
        self.rname='reconnet'
        self.specifics=specifics
        
        self.dims = [input_width,input_height,input_channel]
        self.n = n
        self.m = m
    
    # Sets up directories, the model, sensing, and data loading
    def initialize(self,dset,sensing_method,stage):
        # Create directories and log file
        self.id = u.random_name(16)
        self.model_root, self.logs_root = u.create_dirs(
            self.specifics['model-root'],
            self.specifics['logs-root'],
            self.rname,
            stage,
            dset.name,
            self.specifics['save-name'] if 'save-name' in self.specifics else ('cr' + str(self.dims[0] * self.dims[1] // self.m))
        )
        self.log_file = open(os.path.join(self.logs_root, self.id + '.txt'), 'w')
        self.log_file.write("Params: " + str(self.specifics) + '\n')
        
        # Create model and sensing
        self.device = torch.device(self.specifics['device']) if 'device' in self.specifics else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = u.ReconNet(self.m, self.dims[0], self.dims[1], self.dims[2]).to(self.device)
        self.sensing = sensing_method.to(self.device)
        
        # Attempt to load model if requested
        if stage == 'testing':
            self.load_model()
            self.load_sensing()
        elif stage == 'training':
            resume_training = self.specifics['resume-training'] if 'resume-training' in self.specifics else False
        
            if str(resume_training) == 'True':
                self.load_model()
                self.load_sensing()
            
            self.check_dirs()
        
        # Save sensing matrix
        torch.save(self.sensing.state_dict(), os.path.join(self.model_root, sensing_name))
        
        # Handle other components, i.e. data loading, optimizer, and loss
        self.dataset = dset
        
        self.batch_size = self.specifics['batch-size'] if 'batch-size' in self.specifics else len(self.dataset)
        self.workers = self.specifics['workers'] if 'workers' in self.specifics else 0
        
        if stage == 'testing':
            self.testdataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers
            )
        elif stage == 'training':
            # If user defined validation images, then create validation set
            if 'validation-images' in self.specifics:
                self.validation_images = self.specifics['validation-images']
                
                if isinstance(self.validation_images, float) and self.validation_images < 1:
                    self.validation_images = int(self.validation_images * len(self.dataset))
                
                self.trainset, self.valset = torch.utils.data.random_split(
                    self.dataset,
                    [len(self.dataset) - self.validation_images, self.validation_images],
                    generator=torch.Generator().manual_seed(self.specifics['validation-split-seed'] if 'validation-split-seed' in self.specifics else 2147483647)
                )
                self.valdataloader = torch.utils.data.DataLoader(
                    self.valset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers
                )
            # Otherwise, train on entire dataset (no validation images)
            else:
                self.validation_images = 0
                self.trainset = self.dataset
            
            self.traindataloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers
            )
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.specifics['lr'], betas=self.specifics['betas'])
        else:
            raise NotImplementedError
        self.loss_f=torch.nn.MSELoss(reduction='mean')
    
    # Attempts to load a model from either specific-model-path (if in specifics)
    # or a model from self.model_root (best_model_name if it exists, otherwise a random one)
    def load_model(self):
        if 'specific-model-path' in self.specifics:
            model_path = self.specifics['specific-model-path']
            
            if not os.path.exists(model_path) or not os.path.isfile(model_path):
                raise Exception("No model found at %s"%model_path)
        else:
            models = [name for name in os.listdir(self.model_root) if name.endswith(model_ext)]
            if len(models) > 0:
                if best_model_name in models:
                    model_path = os.path.join(self.model_root, best_model_name)
                else:
                    idx = torch.randint(len(models), (1,)).item()
                    model_path = os.path.join(self.model_root, models[idx])
            else:
                raise Exception("No models found in %s"%self.model_root)
        
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        print("Loaded model from %s"%model_path)
        self.log_file.write("Loaded model from %s\n"%model_path)
    
    # Attempts to load a sensing model from either sensing-path (if in specifics)
    # or a sensing model from self.model_root (sensing_name if it exists, otherwise a random one)
    def load_sensing(self):
        if 'sensing-path' in self.specifics:
            sensing_path = self.specifics['sensing-path']
            
            if not os.path.exists(sensing_path) or not os.path.isfile(sensing_path):
                raise Exception("No sensing model found at %s"%sensing_path)
        else:
            sensors = [name for name in os.listdir(self.model_root) if name.endswith(sensing_ext)]
            if len(sensors) > 0:
                if sensing_name in sensors:
                    sensing_path = os.path.join(self.model_root, sensing_name)
                else:
                    idx = torch.randint(len(models), (1,)).item()
                    sensing_path = os.path.join(self.model_root, sensors[idx])
            else:
                raise Exception("No sensing models found in %s"%self.model_root)
        
        self.sensing.load_state_dict(torch.load(sensing_path, map_location=self.device))
        print("Loaded sensing model from %s"%sensing_path)
        self.log_file.write("Loaded sensing model from %s\n"%sensing_path)
    
    # Check if starting training will overwrite 
    def check_dirs(self):
        if os.path.exists(self.model_root) and len([name for name in os.listdir(self.model_root) if name.endswith(model_ext)]) > 0:
            while True:
                ans = input("WARNING: Training this session will likely overwrite a previous model. Would you like to continue? (Y/N)\n").lower().strip()
                
                if ans == 'y' or ans == 'yes':
                    return
                elif ans == 'n' or ans == 'no':
                    self.log_file.write("Manually exited\n")
                    exit()
    
    # Run the model
    def run(self,stage):
        self.log_file.write("Started %s\n"%time.asctime())
        
        if stage=="training":
            self.train()
        elif stage=="testing":
            self.test()
        else:
            raise NotImplementedError
            
        self.log_file.write("\nEnded %s\n"%time.asctime())
        self.log_file.close()
    
    # Start training loop
    def train(self):
        print("Starting training...")        
        min_loss = np.inf
        start = time.time()
        
        for epoch in range(self.specifics['epochs']):
            avg_loss, avg_psnr, avg_ssim = self.train_epoch(epoch)
            self.log_file.write("\nEpoch %d: Avg. Loss: %f PSNR: %f SSIM: %f, Min. Loss: %f"%(epoch,avg_loss,avg_psnr,avg_ssim,min_loss))

            if self.validation_images > 0:
                val_loss, val_psnr, val_ssim = self.validate()
                self.log_file.write("\nEpoch %d: Val. Loss: %f PSNR: %f SSIM: %f"%(epoch,val_loss,val_psnr,val_ssim))
            else:
                val_loss = avg_loss
            
            if val_loss < min_loss:
                print("Saving new best model: %f > %f"%(min_loss, val_loss))
                min_loss = val_loss
                if os.path.exists(os.path.join(self.model_root, best_model_name)):
                    os.rename(os.path.join(self.model_root, best_model_name), os.path.join(self.model_root, best_model_name + '.old'))
                torch.save(self.net.state_dict(), os.path.join(self.model_root, best_model_name))
            else:
                print("Retained previous best model: %f <= %f"%(min_loss, val_loss))
            
            if 'save-interval' in self.specifics and epoch % self.specifics['save-interval'] == 0:
                print("Saving checkpoint")
                torch.save(self.net.state_dict(), os.path.join(self.model_root, 'epoch%d'%epoch + model_ext))
        
        end = time.time()
        self.log_file.write("\n\nTime elapsed: %f seconds\n"%(end - start))
    
    # Train over entire dataset once
    def train_epoch(self, epoch):
        self.net.train()
        print("\nBeginning epoch %d"%epoch)
        total_loss = 0
        total_items = 0
        total_psnr = 0
        total_ssim = 0
    
        for img in iter(self.traindataloader):
            # Train model
            self.optimizer.zero_grad()
            img = img.to(self.device)
            measurement = self.sensing(img)
            img_hat = self.net(measurement)
            loss = self.loss_f(img_hat, img)
            loss.backward()
            self.optimizer.step()
            
            # Move imgs to CPU for metric calculations
            img = img.cpu()
            img_hat = img_hat.detach().cpu()
            
            # Calculate performance metrics
            batch_loss = loss.item()
            total_loss += batch_loss * img.shape[0]
            
            batch_psnrs = u.compute_psnr(img, img_hat)
            batch_psnr = sum(batch_psnrs)/len(batch_psnrs)
            total_psnr += sum(batch_psnrs)
            
            batch_ssims = u.compute_ssim(img, img_hat)
            batch_ssim = sum(batch_ssims)/len(batch_ssims)
            total_ssim += sum(batch_ssims)
            
            total_items += img.shape[0]
            print("Epoch %d [%d / %d] Loss: %f PSNR: %f SSIM: %f"%(epoch, total_items, len(self.trainset), batch_loss, batch_psnr, batch_ssim))
        
        avg_loss = total_loss / total_items
        avg_psnr = total_psnr / total_items
        avg_ssim = total_ssim / total_items
        
        print("Epoch %d Avg. Loss: %f PSNR: %f SSIM: %f"%(epoch, avg_loss, avg_psnr, avg_ssim))
        return avg_loss, avg_psnr, avg_ssim
        
    def validate(self):
        with torch.no_grad():
            self.net.eval()
            print("\nBeginning validation")
            total_loss = 0
            total_items = 0
            total_psnr = 0
            total_ssim = 0
        
            for img in iter(self.valdataloader):
                # Feed imgs through network
                img = img.to(self.device)
                measurement = self.sensing(img)
                img_hat = self.net(measurement)
                loss = self.loss_f(img_hat, img)
                
                # Move imgs to CPU for metric calculations
                img = img.cpu()
                img_hat = img_hat.detach().cpu()
                
                # Calculate performance metrics
                batch_loss = loss.item()
                total_loss += batch_loss * img.shape[0]
                
                batch_psnrs = u.compute_psnr(img, img_hat)
                batch_psnr = sum(batch_psnrs)/len(batch_psnrs)
                total_psnr += sum(batch_psnrs)
                
                batch_ssims = u.compute_ssim(img, img_hat)
                batch_ssim = sum(batch_ssims)/len(batch_ssims)
                total_ssim += sum(batch_ssims)
                
                total_items += img.shape[0]
            
            avg_loss = total_loss / total_items
            avg_psnr = total_psnr / total_items
            avg_ssim = total_ssim / total_items
            
            print("Validation Loss: %f PSNR: %f SSIM: %f"%(avg_loss, avg_psnr, avg_ssim))
            return avg_loss, avg_psnr, avg_ssim
    
    # Start testing loop
    def test(self):
        print("Starting testing...")
        with torch.no_grad():
            self.net.eval()
            
            val_psnrs = []
            val_ssims = []
            total_items = 0
            
            # Figure out number of images to save from dataset size
            if len(self.dataset) >= 16:
                saved_items = 16
            elif len(self.dataset) >= 9:
                saved_items = 9
            elif len(self.dataset) >= 4:
                saved_items = 4
            elif len(self.dataset) >= 1:
                saved_items = 1
            
            saved_img = []
            saved_img_hat = []
            times = 0
            
            for img in iter(self.testdataloader):
                # Feed imgs through network
                img = img.to(self.device)
                measurement = self.sensing(img)
                start = time.time()
                img_hat = self.net(measurement)
                end = time.time()
                times += end - start
                
                # Move imgs to CPU for future usage
                img = img.cpu()
                img_hat = img_hat.detach().cpu()
                
                # Calculate performance metrics
                batch_psnrs = u.compute_psnr(img, img_hat)
                batch_psnr = sum(batch_psnrs)/len(batch_psnrs)
                val_psnrs += batch_psnrs
                
                batch_ssims = u.compute_ssim(img, img_hat)
                batch_ssim = sum(batch_ssims)/len(batch_ssims)
                val_ssims += batch_ssims
                
                # Save images from first batch
                if len(saved_img) < saved_items:
                    saved_img.extend(img[:saved_items - len(saved_img)])
                    saved_img_hat.extend(img_hat[:saved_items - len(saved_img_hat)])
                
                total_items += img.shape[0]
                print("[%d / %d] PSNR: %f SSIM: %f"%(total_items, len(self.dataset), batch_psnr, batch_ssim))
                self.log_file.write("[%d / %d] PSNR: %f SSIM: %f\n"%(total_items, len(self.dataset), batch_psnr, batch_ssim))
            
            val_psnr = sum(val_psnrs) / len(val_psnrs)
            val_ssim = sum(val_ssims) / len(val_ssims)
            avg_time = times / total_items
            
            print("Avg. PSNR: %f SSIM: %f Reconstruction Speed: %f"%(val_psnr, val_ssim, avg_time))
            self.log_file.write("\nAvg. PSNR: %f SSIM: %f Reconstruction Speed: %f\n"%(val_psnr, val_ssim, avg_time))
            
            u.save_imgs(saved_img, saved_img_hat, os.path.join(self.logs_root, self.id + '.png'))

