# ReconNet
### Description
This implementation of CSGM is based off of the original implementation at https://github.com/AshishBora/csgm . The various scripts that were used to run the algorithm with various parameters were  


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
* m: The number of measurements to take for the entire image (all 3 channels included, NOT per channel). For more details, view `main.py`
* n: The total size of the image in pixels
* specifics: The specific parameters for the method. Presets have been defined for convenience, and the following 4 presets were used to evaluate the method:
(Note: only parameters that were changed during the evaluation are commented here. Parameters that remained constant are not documented here, for more information please refer to the original implementation)
  dcganTrain = { //used for training every dataset except for MNIST
    'epoch': 300,
    'learning_rate': 0.0002,
    'beta1':0.5,
    'train_size': np.inf,
    'batch_size':64, //batch size used for training
    'input_height':32, //input dimension of the image
    'input_width':None,
    'output_height':32,
    'output_width':None,
    'dataset':"cifar10_gray/train", //path to the training set within the /data subdirectory
    'input_fname_pattern':"*.png", //file extension of the images
    'data_dir':'./data', //root directory for the data
    'out_dir':'./models', //root directory where the model will be saved
    'out_name':'cifar_10_gray', //name of the folder within the model root directory where the model will be saved
    'checkpoint_dir': 'checkpoint',
    'sample_dir':'samples',
    'train':True,
    'crop':True,
    'visualize':False,
    'export':False,
    'freeze':False,
    'max_to_keep':None,
    'sample_freq':200, //Frequency of 
    'ckpt_freq':200,
    'z_dim': 100,
    'z_dist':'uniform_signed',
    'G_img_sum': False
  }

  vaeTrain = { //used for training on MNIST
    'num-samples': 60000, //number of training images
    'train_data': './data/mnist_benchmark/train/*.png', //path to training data
    'learning-rate': 0.001,
    'batch-size': 100, //batch size for training
    'input-size': 32,
    'training-epochs': 200,
    'summary-epochs': 1,
    'ckpt_epoch': 5,
  }











  * save-name: The custom name, not path, of the folder in which to save the model. Defaults to cr{n // m where // denotes floor division}
  * test-root: The path to the folder containing the testing images
  * train-root: The path to the folder containing the training images
  * model-root: The path to the folder in which to save all models. The file structure will be constructed as follows: {model-root} > reconnet > {dataset} > {save-name} > model/sensing files
  * logs-root: The path to the folder in which to save all logs. The file structure will be constructed as follows: {logs-root} > reconnet > {stage} > {dataset} > {save-name} > log files
  * epochs: The number of epochs for which to train the model
  * lr: The learning rate for the Adam optimizer
  * betas: The beta values for the Adam optimizer
  * batch-size: The batch size to use for training/testing
  * specific-model-path: The specific model to load for training/testing
  * sensing-path: The sensing method to load for training/testing
  * resume-training: Whether the model/sensing method should be loaded from the specific-model-path and sensing-path to resume training. Ignored when stage != 'training'
  * save-interval: How often to save a checkpoint while training in terms of epochs
  * max-images: The maximum number of images to use for training/testing
  * validation-images: Either an integer or a float in the range \[0,1\]. If an integer, will use that many training images for validation. If a float, will use that portion of the training images for validation.
  * validation-split-seed: The seed to use for randomly splitting the train set into validation/training. Recommended to set or leave as default in order to ensure consistency between training sessions
  * workers: The number of workers to use for loading the dataset. Recommended between 0 to 2
  * device: The pytorch device on which to train/test on. To run on CPU, set it to 'cpu'. To run on gpu, set it to 'cuda:N' where N is the index of the GPU to use, starting from 0
  * validate-data: Whether to check if data can be properly transformed before adding it to the dataset. This process is slow, so it is recommended to clean the data beforehand and keep this option off
