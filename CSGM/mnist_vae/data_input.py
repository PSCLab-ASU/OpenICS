# Get input
import glob
import CSGM.dcgan.dcgan_utils as dcgan_utils
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def mnist_data(hparams, num_batches):
   # mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    """Create input tensors"""
    image_paths = glob.glob(hparams.train_data)

    image_paths.sort() 


    images = [dcgan_utils.get_image(image_path,hparams.input_size,True,hparams.input_size,True) for image_path in image_paths]#images = [dcgan_utils.get_image(image_path, image_size) for image_path in image_paths]
    images = [image.reshape([hparams.input_size*hparams.input_size*1])/2+0.5 for image in images]
    images = np.array(images)
    image_batches = [images[i*hparams.batch_size:(i+1)*hparams.batch_size] for i in range(num_batches)]
    image_batches = np.array(image_batches)
    
    return image_batches
