# CSGAN
### Description
The CSGAN implementation within the framework is based off the original implementation at https://github.com/deepmind/deepmind-research/tree/master/cs_gan. To reproduce the pretrained results, make sure to copy the sensing matrix .npy files into the method's directory.

The method may be directly ran by modifying and calling `main.py`. Otherwise, the method may be ran by calling the main function from another file and passing in custom arguments.

### Parameters:
* reconstruction: Currently only supports 'CSGAN'
* sensing: Possible values are: random sensing: 'sensing_matrix' or a neural network: "NN_MLP" (used for MNIST) or "NN_DCGAN" (used for all other datasets)
* stage: Either 'training' for training a model or 'testing' for evaluating a model's performance
* default: Utilizes pre-determined parameters for training
* dataset: The name of the dataset being trained/tested on
* input_channel: The number of channels in each image
* input_width: The width of each image
* input_height: The height of each image
* n: Input size of the images (width*height*channels)
* m: The number of measurements to take per image (including all channels, NOT perc channel)
* specifics: The specific parameters for the method.
  * num_training_iterations: Number of iterations (1 iteration is 1 batch) to run the method for
  * learned_sensing_parameters: Whether or not to add the sensing parameters (of a random sensing matrix or a Discriminator network) to the list of parameters of the optimizer
  * batch_size: Number of images in one batch
  * num_latents: Size of latent space and tensor fed into the generator
  * summary_every_step: The interval at which to log debug ops.
  * save_every_step: The interval at which to save model dicts
  * val_every_step: The interval for running validation
  * num_z_iters: The number of latent optimisation steps. It falls back to vanilla GAN when num_z_iters is set to 0.
  * z_step_size: Step size for latent optimisation.
  * lr: Learning rate of the optimizers
  * z_project_method: The method to project z
  * output_file: Output file for latest model checkpoint
  * output_file_best: Output file for best model evaluated on the validation set
  * log_file: Output file for saving the console log output
  * generator: Either MLP (originally used for MNIST) or DCGAN (originally used for CelebA, for the benchmarks all datasets except for MNIST use DCGAN)

  * copy_dataset_source_folder: If the dataset folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
  * dataset: train/val/test sets will be loaded from here by default. If the folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
  * n_Val_Images: Number of validation images for the dataset split
  
