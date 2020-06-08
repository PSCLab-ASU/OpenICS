import torch
import reconstruction_methods as rms
import utils
import sensing_methods as sms
import numpy as np

def main(sensing,reconstruction,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        sensing = "Gaussian"
        reconstruction = "CSGAN"
        stage = stage
        dataset = '.\exampleLocation' # location of dataset

        # dset = torch.load(dataset)
        # cifar10
        # batch_size, height, width, n_channels = dset.shape

        input_channel = 1 # n_channels
        input_width = 28 # width
        input_height = 28 # height

        specifics = {
            'num_training_iterations': 200000,
            'batch_size': 64,
            'num_latents': 128,
            'summary_every_step': 1000, # The interval at which to log debug ops.
            'image_metrics_every_step': 2000, # The interval at which to log (expensive) image metrics.
            'export_every': 10, # The interval at which to export samples.
            'num_eval_samples': 10000,
            'dataset': 'cifar', # 'The dataset used for learning (cifar|mnist.'
            'num_z_iters': 3, # 'The number of latent optimisation steps. It falls back to vanilla GAN when num_z_iters is set to 0.'
            'z_step_size': 0.01, # 'Step size for latent optimisation.'
            'z_project_method': 'norm',
            'output_dir': '/tmp/cs_gan/gan',
            'disc_lr': 2e-4,
            'gen_lr': 2e-4,
            'run_real_data_metrics': False,
            'run_sample_metrics': True,
            'input_channel': input_channel
        }


    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage=stage) #default mnist
    sensing_method=sms.sensing_method(sensing, specifics, m, n)
    reconstruction_method=rms.reconstruction_method(reconstruction,specifics)
    # put result of the parameters into specifics.
    reconstruction_method.initialize(dset,sensing_method,specifics)
    reconstruction_method.run(stage)
        

if __name__ == "__main__":
    main(
        "Gaussian",
        "CSGAN",
        "training",
        "True",
        "./TrainingData/TrainingData_patch",
        1,
        40,
        40,
        40*40*1,
        int(np.round(.2*40*40*1)),
        {
            "tie_weights": False,
            "filter_height": 3,
            "filter_width": 3,
            "num_filters": 64,
            "n_DnCNN_layers": 16,
            "max_n_DAMP_layers": 10,
            "BATCH_SIZE": 1, # Using a batch size larger than 1 will hurt the denoiser by denoiser trained network because it will use an average noise level, rather than a noise level specific to each image
            "n_Test_Images": 5,
            "sampling_rate_test": .2,  # The sampling rate used for testing
            "sampling_rate_train": .2,  # The sampling rate that was used for training
            "sigma_w": 0.,
            "init_mu": 0,
            "init_sigma": 0.1
        },
    )