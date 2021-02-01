# LAPRAN
### Description
The LAPRAN implementation within the framework is based off the original implementation at https://github.com/PSCLab-ASU/LAPRAN-PyTorch.

The method may be directly ran by modifying and calling `main.py`, which contains the specifics at the top of the file and other parameters at the bottom. Otherwise, the method may be ran by calling the main function from another file and passing in custom arguments.

### Variables:
* sensing: Currently only supports 'random-sensing'
* reconstruction: Currently only supports 'LAPRAN'
* stage: Either 'training' for training a model or 'testing' for evaluating a model's performance
* default: Utilizes pre-determined parameters for training
* dataset: The name of the dataset being trained/tested on
* input_channel: The number of channels in each image
* input_width: The width of each image
* input_height: The height of each image
* m: The number of measurements to take per channel
* n: Size of the images per channel (width*height)
* specifics: The specific parameters for the method. There are separately predefined parameters for training and testing.
  * Testing
    * model: The model to use. Currently only "adaptiveCS_resnet_wy_ifusion_ufirst" is supported
    * dataset: The dataset to use
    * batch_size: The number of images in a batch
    * lr: Learning rate for the optimizers
    * cuda: Set to true to enable CUDA training
    * ngpu: Number of GPUs to use
    * seed: Random seed to use
    * gpu: Preferred GPU to use
    * outf: Folder to load model checkpoints
    * w-loss: Penalty for the MSE and BCE loss
    
    * test-root: The path to the folder containing the testing images
    * train-root: The path to the folder containing the training images
   * Training
     * model: The model to use. Currently only "adaptiveCS_resnet_wy_ifusion_ufirst" is supported
     * batch_size: The number of images in a batch
     * epochs: Number of epochs to train
     * lr: Learning rate for the optimizers
     * cuda: Set to true to enable CUDA training
     * ngpu: Number of GPUs to use
     * seed: Random seed to use
     * log-interval: Number of batches to wait before logging training status
     * gpu: Preferred GPU to use
     * outf: Folder to output images and model checkpoints
     * w-loss: Penalty for the MSE and BCE loss
     * stage: Stage under training
     * copy_dataset_source_folder: If the dataset folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
     * dataset: train/val/test sets will be loaded from here by default (it is assumed that the folder is within /data folder of root). If the folder does not exist, the program will attempt to create the dataset splits from the copy_dataset_source_folder, where it assumes a test and train subfolder is present
     * n_Val_Images: Number of validation images for the dataset split

    
