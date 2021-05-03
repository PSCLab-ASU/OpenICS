import torch
import CSGAN.reconstruction_methods as rms
import CSGAN.utils as utils
import CSGAN.sensing_methods as sms
import numpy as np

def main(reconstruction,sensing,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        reconstruction= "CSGAN"
        sensing = "sensing_matrix"
        stage = "training"
        dataset = "mnist"
        input_channel = 1
        input_width = 32
        input_height = 32
        n = 32*32*1
        m = 32


    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics)
    sensing_method= sms.sensing_method(sensing,None, None, None, None)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing, specifics)

    reconstruction_method.initialize(dset,sensing_method,stage,specifics)
    reconstruction_method.run(stage)
        

if __name__ == "__main__":
    specifics = {
            'num_training_iterations': 1000000, #Number of iterations (1 iteration is 1 batch) to run the method for
            'learned_sensing_parameters': False, #Whether or not to add the sensing parameters (of a random sensing matrix or a Discriminator network) to the list of parameters of the optimizer
            'batch_size': 1, #Number of images in one batch
            'num_latents': 100, #Size of latent space and tensor fed into the generator
            'summary_every_step': 100, # The interval at which to log debug ops.
            'save_every_step': 1000, # The interval at which to save model dicts
            'val_every_step' :500 , #The interval for running validation
            'num_z_iters': 3, #The number of latent optimisation steps. It falls back to vanilla GAN when num_z_iters is set to 0.
            'z_step_size': 0.01, #Step size for latent optimisation.
            'lr': 1e-4, #Learning rate of the optimizers
            'z_project_method': 'norm', #The method to project z
            'output_file': 'saved_models/mnist_32_sensing_matrix_notlearned.pth',  #Output file for latest model checkpoint
            'output_file_best': 'saved_models/mnist_32_sensing_matrix_notlearned_BEST.pth',  #Output file for best model evaluated on the validation set
            'log_file': 'saved_models/mnist_32_sensing_matrix_notlearned.txt', #Output file for saving the console log output
            'generator': "MLP", #Either MLP (originally used for MNIST) or DCGAN (originally used for CelebA, for the benchmarks all datasets except for MNIST use DCGAN)

            'copy_dataset_source_folder': "/storage-t1/database/cs-framework-database/mnist", #If the dataset folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
            'dataset': './data/mnist',  #train/val/test sets will be loaded from here by default. If the folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
            'n_Val_Images': 10000 #Number of validation images for the dataset split
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
        m = 32,
        specifics = specifics
    )