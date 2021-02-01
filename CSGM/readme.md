# CSGM
### Description
This implementation of CSGM is based off of the original implementation at https://github.com/AshishBora/csgm. The main script parameters that were used to run the algorithm with various parameters are defined as


The method may be directly ran by modifying and calling `main.py`, which contains the specifics at the top of the file and other parameters at the bottom. Otherwise, the method may be ran by calling the main function from another file and passing in custom arguments.

### Parameters:
* sensing: Currently only supports 'gaussian'
* reconstruction: Currently only supports 'csgm'
* stage: Either 'training' for training a model or 'testing' for evaluating a model's performance
* default: Utilizes pre-determined parameters for training
* dataset: Only used to determine the generator being used and should be set to either 'mnist' for MNIST, and 'celebA' for all other datasets. To specify the datasets, see presets.py 
* input_channel: The number of channels in each image
* input_width: The width of each image
* input_height: The height of each image
* m: The number of measurements to take for the entire image (all 3 channels included, NOT per channel).
* n: The total size of the image in pixels
* specifics: The specific parameters for the method. Presets have been defined for convenience, and the following 4 presets were used to evaluate the method:
vaeTrain = { # Used for training for MNIST
    'num-samples': 60000, # Number of training images in the dataset
    'train_data': './data/mnist_benchmark/train/*.png', #Path and file extension to the training set
    'learning-rate': 0.001, # Learning rate of the optimizer
    'batch-size': 100, # Number of images in for a batch
    'input-size': 32, # Dimensionality of the image (image width, assuming it is a square image)
    'training-epochs': 200, # Number of training epochs 
    'summary-epochs': 1, 
    'ckpt_epoch': 5, # Interval between epochs to save checkpoints
}
dcganTrain = { # Used for training on every dataset except for MNIST
    'epoch': 300, # Number of epochs to train for
    'learning_rate': 0.0002, # Learning rate for the adam optimizer
    'beta1':0.5, # Momentum term of the adam optimizer
    'train_size': np.inf, # The size of train images
    'batch_size':64, # Number of images in a batch
    'input_height':32, # Input image height (will be center cropped)
    'input_width':None, # Input image width (will be center cropped). If None, same value as input_height.
    'output_height':32, # Height of the output images to produce
    'output_width':None, # The width of the output images to produce. If None, same value as output_height 
    'dataset':"cifar10_gray/train", # Subfolder of the dataset within "data_dir"
    'input_fname_pattern':"*.png", # File extension for the data
    'data_dir':'./data', # Main directory where the datasets are stored
    'out_dir':'./models', # Root output directory to save model checkpoints and logs
    'out_name':'cifar_10_gray', # Subfolder for the current model within the out_dir directory
    'checkpoint_dir': 'checkpoint', # Checkpoint directory name within the out_name directory
    'sample_dir':'samples', # Sample directory name within the out_name directory
    'train':True, # Should be set to True
    'crop':True, # Decides whether to crop the input
    'visualize':False, # True for visualizing, False for nothing
    'export':False, #True for exporting with new batch size
    'freeze':False, #True for exporting with new batch size
    'max_to_keep':None, #maximum number of checkpoints to keep
    'sample_freq':200, # Frequency of exporting samples (in terms of batches)
    'ckpt_freq':200, # Frequency of saving checkpoints (in terms of batches)
    'z_dim': 100, # Dimensionality of latent space and tensor fed into the generator
    'z_dist':'uniform_signed', #'normal01' or 'uniform_unsigned' or uniform_signed
    'G_img_sum': False # Save generator image summaries in log
}
dcganWithRegCustom = { # Used for testing every dataset except for MNIST
            'pretrained-model-dir': './models/celebA_64_64/', # Path to the pretrained model directory
            'test_data': './data/celebAtest/*.jpg', # Path and file extension of the test data
            'input-type': "full-input", # For evaluation, this should be set to full-input. See the original implementation for more details.
            'num-input-images' : 867, # The number of images to use from the test set
            'batch-size': 64, # The number of images in a batch that should be used for testing
            'noise-std': 0.01, # Std dev of noise

            'model-types': ["dcgan"], # model(s) used for estimation
            'mloss1_weight':0.0, #L1 measurement loss weight
            'mloss2_weight':1.0, #L2 measurement loss weight
            'zprior_weight':0.001, #weight on z prior
            'dloss1_weight':1.0, #-log(D(G(z)))
            'dloss2_weight':0, #log(1-D(G(z)))

            
            'optimizer-type': "adam", #Optimizer type
            'learning-rate': 0.1,
            'momentum':0.9, 
            'max-update-iter':500, #Maximum updates to z
            'num-random-restarts': 10, #Number of random restarts
            'lmbd': 0, #Regularization parameter for LASSO

            'save-stats' : False, #Whether to save estimated images
            'print-stats' : True, #Whether to print statistics
}
vaeWithRegCustom = { # Used for testing MNIST
            'pretrained-model-dir': './models/mnist-vae/',# Path to the pretrained model directory
            'test_data': './data/mnist_benchmark/test/*.png',# Path and file extension of the test data
            'input-type': "full-input",# For evaluation, this should be set to full-input. See the original implementation for more details.
            'num-input-images' : 10000,# The number of images to use from the test set
            'batch-size': 100,# The number of images in a batch that should be used for testing
            'noise-std': 0.1,# Std dev of noise

            'model-types': ["vae"], # model(s) used for estimation
            'mloss1_weight':0.0,#L1 measurement loss weight
            'mloss2_weight':1.0,#L2 measurement loss weight
            'zprior_weight':0.1,#weight on z prior
            'dloss1_weight':0,#-log(D(G(z)))
            'dloss2_weight':0,#log(1-D(G(z)))

            
            'optimizer-type': "adam",#Optimizer type
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':1000,#Maximum updates to z
            'num-random-restarts': 10, #Number of random restarts
            'lmbd': 0,#Regularization parameter for LASSO

            'save-stats' : False,#Whether to save estimated images
            'print-stats' : True,#Whether to print statistics
}
