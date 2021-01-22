# ReconNet
### Description
The ReconNet implementation within the framework is based off the original method described in this [paper](https://arxiv.org/pdf/1601.06892.pdf).

The method may be directly ran by modifying and calling `main.py`, which contains the specifics at the top of the file and other parameters at the bottom. Otherwise, the method may be ran by calling the main function from another file and passing in custom arguments.

### Variables:
* sensing: Currently only supports 'random'
* reconstruction: Currently only supports 'reconnet'
* stage: Either 'training' for training a model or 'testing' for evaluating a model's performance
* default: Utilizes pre-determined parameters for training
* dataset: The name of the dataset being trained/tested on
* input_channel: The number of channels in each image
* input_width: The width of each image
* input_height: The height of each image
* m: The number of measurements to take per channel. For more details, view `main.py`
* n: The total size of the image in pixels
* specifics: The specific parameters for the method. Parameters are detailed below.
  * test-root: The path to the folder containing the testing images
  * train-root: The path to the folder containing the training images
  * model-root: The path to the folder in which to save all models. The file structure will be as follows: {model-root} > reconnet > {dataset} > cr{n // m where // denotes floor division} > model/sensing files
  * logs-root: The path to the folder in which to save all logs. The file structure will be as follows: {logs-root} > reconnet > {stage} > {dataset} > cr{n // m} > log files
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
