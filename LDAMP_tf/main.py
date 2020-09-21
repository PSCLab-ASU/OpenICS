#TODO changed the import tensorflow as tf line to two lines as seen below
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
import numpy as np
import reconstruction_methods as rms
import utils
import sensing_methods as sms
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def main(sensing,reconstruction,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default==True:
        sensing = sensing
        reconstruction = reconstruction
        stage = stage
        dataset = dataset

        input_channel = input_channel
        input_width = input_width
        input_height = input_height

        # needed to calculate matrices sizes
        sampling_rate = .2
        sigma_w = 1. / 255.  # Noise std

        n = input_channel * input_height * input_width
        m = int(np.round(sampling_rate * n))
        specifics = {
        'channel_img': input_channel,
        'width_img': input_width,
        'height_img': input_height,
        'sampling_rate': sampling_rate,
        'sigma_w': sigma_w,  # Noise std
        'n': n,
        'm': m,
        'alg': reconstruction,
        'tie_weights': False, # if true set LayerByLayer to False
        'filter_height': 3,
        'filter_width': 3,
        'num_filters': 64,
        'n_DnCNN_layers': 16,
        'max_n_DAMP_layers': 1,  # Unless FLAGS.start_layer is set to this value or LayerbyLayer=false, the code will sequentially train larger and larger networks end-to-end.

        ## Training Parameters
        'start_layer': 1,
        'max_Epoch_Fails': 3,  # How many training epochs to run without improvement in the validation error
        'ResumeTraining': False,  # Load weights from a network you've already trained a little
        'LayerbyLayer': True,  # Train only the last layer of the network
        'DenoiserbyDenoiser': False, # if this is true, overrides other two
        'learning_rates': [0.001, 0.0001],  # , 0.00001]
        'EPOCHS': 50,
        'n_Train_Images': 128 * 1600,  # 128*3000
        'n_Val_Images': 10000,  # 10000#Must be less than 21504
        'BATCH_SIZE': 128,
        'InitWeightsMethod': 'smaller_net',
        'loss_func': 'MSE',
        'init_mu': 0,
        'init_sigma': 0.1,
        'mode': sensing,
        'stage': stage,
        'validation_patch': '/home/mkweste1/LDAMP_TensorFlow/Data/ValidationData_patch',
        'training_patch': '/home/mkweste1/LDAMP_TensorFlow/Data/TrainingData_patch'
        }

    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage) # unused, just used to pass in
    sensing_method=sms.sensing_method(reconstruction,specifics) # unused, just used to pass in
    reconstruction_method=rms.reconstruction_method(dset, reconstruction,specifics)
    # put result of the parameters into specifics.
    reconstruction_method.initialize(dset,sensing_method)
    reconstruction_method.run()

if __name__ == "__main__":
    main(
        'gaussian', # sensing type
        "DAMP", #put DAMP instead of LDAMP
        "training", # stage
        False, # default
        '/home/user/mkweste1/LDAMP final/Data/TrainingData/TrainingData_patch50.npy', # dataset
        # "/home/user/mkweste1/LDAMP final/Data/TrainingData/StandardTestData_256Res.npy",
        1, # input channels
        50, # input width
        50, # input height
        320,  # m 320
        50 * 50 * 1, # n 40x40x1
        {
            'channel_img': 1,
            'width_img': 50,
            'height_img': 50,
            'sampling_rate': .2,
            'sigma_w': 1. / 255.,  # Noise std
            'n': 1 * 50 * 50,
            'm': int(np.round(.2 * 1*50*50)),
            'alg': 'DAMP',
            'tie_weights': False,  # if true set LayerByLayer to False
            'filter_height': 3,
            'filter_width': 3,
            'num_filters': 64,
            'n_DnCNN_layers': 16,
            'max_n_DAMP_layers': 1,
            # Unless FLAGS.start_layer is set to this value or LayerbyLayer=false, the code will sequentially train larger and larger networks end-to-end.

            ## Training Parameters
            'start_layer': 1,
            'max_Epoch_Fails': 3,  # How many training epochs to run without improvement in the validation error
            'ResumeTraining': False,  # Load weights from a network you've already trained a little
            'LayerbyLayer': True,  # Train only the last layer of the network
            'DenoiserbyDenoiser': True,  # if this is true, overrides other two
            'sigma_w_min': .25,
            'sigma_w_max': .25,
            'learning_rates': [0.001, 0.0001],  # , 0.00001]
            'EPOCHS': 50,
            'n_Train_Images': 128 * 1600,  # 128*3000
            'n_Val_Images': 10000,  # 10000#Must be less than 21504
            'BATCH_SIZE': 128,
            'InitWeightsMethod': 'layer_by_layer', #Options are random, denoiser, smaller net, and layer_by_layer.
            'loss_func': 'MSE',
            'init_mu': 0,
            'init_sigma': 0.1,
            'mode': 'gaussian',
            'stage': "training",
            'validation_patch': '/home/mkweste1/LDAMP_TensorFlow/Data/ValidationData_patch',
            'training_patch': '/home/mkweste1/LDAMP_TensorFlow/Data/TrainingData_patch'
        },
    )