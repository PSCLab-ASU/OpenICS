import LAPRAN.reconstruction_methods as rms
import LAPRAN.utils as utils
import LAPRAN.sensing_methods as sms
import argparse
testSpecifics = {
            "model": 'adaptiveCS_resnet_wy_ifusion_ufirst', #The model to use. Currently only "adaptiveCS_resnet_wy_ifusion_ufirst" is supported
            "dataset": "mnist", #The dataset to use
            "batch_size": 1, #The number of images in a batch
            "test_batch_size": 1000,
            "epochs": 100,
            "lr": 2e-4, # Learning rate for the optimizers
            "momentum": 0.5,
            "cuda":True, #Set to true to enable CUDA training
            "ngpu": 1, #Number of GPUs to use
            "seed": 1, #Random seet to use
            "log-interval": 100,
            "layers-gan": 3,
            "gpu": 0, #Preferred GPU to use
            "outf": './results', #Folder to load model checkpoints
            "w-loss": 0.01, #Penalty for the MSE and BCE loss
            "stage": 1, 
            "transfer": False
            }
trainSpecifics = {
            "model": 'adaptiveCS_resnet_wy_ifusion_ufirst', #The model to use. Currently only "adaptiveCS_resnet_wy_ifusion_ufirst" is supported
            "batch_size": 128,#The number of images in a batch
            "test_batch_size": 1000,
            "epochs": 100, #Number of training epochs
            "lr": 2e-4,# Learning rate for the optimizers
            "momentum": 0.5,
            "cuda":True,#Set to true to enable CUDA training
            "ngpu": 1,#Number of GPUs to use
            "seed": 2,#Random seet to use
            "log-interval": 100,
            "layers-gan": 3,
            "gpu": 0,
            "outf": './results',#Folder to save model checkpoints
            "w-loss": 0.01,#Penalty for the MSE and BCE loss
            "stage": 1, #stage under training
            "transfer": False,

            'copy_dataset_source_folder': 'C:/Users/PSI497/Desktop/PSClab/CSGM/data/mnist',
            'dataset': 'mnist',  #train/val/test sets will be loaded from here by default. If the folder does not exist, the program will attempt to create the dataset from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
            'n_Val_Images': 10000 #number of validation images for the dataset split
            }
            

def main(reconstruction, sensing = "random_sensing",stage = "testing",default = "True",dataset = "cifar10",input_channel=3,input_width=64,input_height=64,n = 64**2, m = 64**2//10 ,specifics = None):
    if default=="True":
        sensing = "random_sensing"
        reconstruction = "LAPRAN"
        stage = "training"
        dataset = "mnist"
        input_channel = 1
        input_width = 32
        input_height = 32
        n = 32**2
        m = n//32
        specifics = trainSpecifics
    
    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage, specifics)
    sensing_method=sms.sensing_method(sensing,n,m,input_width,input_channel)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n, specifics)
    reconstruction_method.initialize(dset,sensing_method,stage, specifics)
    reconstruction_method.run(stage, specifics)
        
if __name__ == '__main__':
    main("LAPRAN")        