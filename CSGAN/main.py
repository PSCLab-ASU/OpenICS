import torch
import reconstruction_methods as rms
import utils
import sensing_methods as sms
import numpy as np

def main(reconstruction,sensing,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        sensing = "Gaussian"
        reconstruction = "CSGAN"
        stage = stage
        dataset = dataset # '.\exampleLocation' # location of dataset

        # dset = torch.load(dataset)
        # cifar10
        # batch_size, height, width, n_channels = dset.shape

        input_channel = 1 # n_channels
        input_width = 28 # width
        input_height = 28 # height


    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics)
    sensing_method= sms.sensing_method(sensing,None, None, None, None)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing, specifics)

    reconstruction_method.initialize(dset,sensing_method,stage,specifics)
    reconstruction_method.run(stage)
        

if __name__ == "__main__":
    specifics = {
            'num_training_iterations': 1000000,
            'learned_sensing_parameters': False, #Whether or not to add the sensing parameters (of a random sensing matrix or a Discriminator network) to the list of parameters of the optimizer
            'batch_size': 64,
            'num_latents': 100,
            'summary_every_step': 100, # The interval at which to log debug ops.
            'save_every_step': 1000, # The interval at which to save model dicts
            'val_every_step' :500 , #interval for validation
            'num_z_iters': 3, # 'The number of latent optimisation steps. It falls back to vanilla GAN when num_z_iters is set to 0.'
            'z_step_size': 0.01, # 'Step size for latent optimisation.'
            'lr': 1e-4,
            'z_project_method': 'norm', #The method to project z
            'output_file': 'saved_models/mnist_16_sensing_matrix_notlearned.pth',  #output file for latest model checkpoint
            'output_file_best': 'saved_models/mnist_16_sensing_matrix_notlearned_BEST.pth',  #output file for best model evaluated on the validation set
            'log_file': 'saved_models/mnist_16_sensing_matrix_notlearned.txt',
            'generator': "DCGAN", #should be either MLP (originally used for MNIST) or DCGAN (originally used for CelebA)

            'copy_dataset_source_folder': "/storage-t1/database/cs-framework-database/mnist",
            'dataset': './data/mnist',  #train/val/test sets will be loaded from here by default. If the folder does not exist, the program will attempt to create the dataset from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
            'n_Val_Images': 10000 #number of validation images for the dataset split
        }
    main(
        reconstruction= "CSGAN",
        sensing = "sensing_matrix", #Can be either sensing_matrix or a neural network (NN_MLP or NN_DCGAN)
        stage = "training",
        default = "False",
        dataset = "mnist",
        input_channel = 1,
        input_width = 32,
        input_height = 32,
        n = 32*32*1,
        m = 10,
        specifics = specifics
    )