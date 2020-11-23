#TODO tensorflow version 2.X migration code changed the import tensorflow as tf line to two lines as seen below
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# import tensorflow as tf
import numpy as np
import reconstruction_methods as rms
import utils
import sensing_methods as sms
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

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
            'channel_img': 1,
            'width_img': input_width,
            'height_img': input_height,
            'n': n,
            'm': m,
            'sampling_rate': sampling_rate,
            'alg': 'DAMP',
            'tie_weights': False,  # if true set LayerByLayer to False
            'filter_height': 3,
            'filter_width': 3,
            'num_filters': 64,
            'n_DnCNN_layers': 16,
            'max_n_DAMP_layers': 10,
            # Unless FLAGS.start_layer is set to this value or LayerbyLayer=false, the code will sequentially train larger and larger networks end-to-end.

            ## Training Parameters
            'start_layer': 1,
            'max_Epoch_Fails': 3,  # How many training epochs to run without improvement in the validation error
            'ResumeTraining': False,  # Load weights from a network you've already trained a little
            'LayerbyLayer': True,  # Train only the last layer of the network
            'DenoiserbyDenoiser': True,  # if this is true, overrides other two
            'sigma_w_min': 25,
            'sigma_w_max': 25,
            'sigma_w': 25. / 255.,  # Noise std
            'learning_rates': [0.001, 0.0001],  # , 0.00001]
            'EPOCHS': 50,
            'n_Train_Images': 128 * 1600,  # 128*3000
            'n_Val_Images': 10000,  # 10000
            'BATCH_SIZE': 128, # 128 for training, 1 for testing
            'InitWeightsMethod': 'smaller net', #Options are random, denoiser, smaller net, and layer_by_layer.
            'loss_func': 'MSE',
            'init_mu': 0,
            'init_sigma': 0.1,
            'mode': 'gaussian',

            'validation_patch': './Data/ValidationData_patch40.npy',
            'training_patch': './Data/TrainingData_patch40.npy',
            'testing_patch': './Data/StandardTestData_256Res.npy',
            'use_separate_val_patch': True,
            'create_new_dataset': False,
            'new_data': '/storage-t1/database/bigset/train/data/*.bmp',
            'dataset_custom_name': 'bigset_train_data',
            'create_new_testpatch': False,
            'new_test_data': '/storage-t1/database/bigset/test/*.bmp',
            'testset_custom_name': 'bigset_test'
        }

    if(not(n == input_channel * input_height * input_width)):
        raise Exception("n must equal input_channel * input_height * input_width")
    if(not(m == int(np.round(specifics['sampling_rate'] * input_channel * input_height * input_width)))):
        raise Exception("m must equal int(np.round(sampling_rate * input_channel * input_height * input_width))")

    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage, specifics)
    sensing_method=sms.sensing_method(reconstruction,specifics) # unused, just used to pass in
    reconstruction_method=rms.reconstruction_method(dset, sensing,specifics)
    reconstruction_method.initialize(dset,sensing_method, stage)
    reconstruction_method.run()

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    sampling_rate = .2
    input_channel = 1
    input_width = 64 # 40
    input_height = 64 # 40
    n = input_channel * input_height * input_width
    m = int(np.round(sampling_rate * input_channel * input_height * input_width))

    # sensing type is only for layer-by-layer and end-to-end
    sensingtype = 'gaussian' #'gaussian', 'complex-gaussian', 'coded-diffraction', (unsupported)'Fast-JL'
    main(
        sensingtype, # sensing type
        "DAMP", #put DAMP instead of LDAMP
        "training", # stage
        False, # default, switch to false if want to edit parameters below
        '', # dataset is not used
        input_channel, # input channels
        input_width, # input width
        input_height, # input height
        m,  # m 320
        n, # n
        {
            'channel_img': input_channel,
            'width_img': input_width,
            'height_img': input_height,
            'n': n,
            'm': m,
            'sampling_rate': sampling_rate,
            'alg': 'DAMP',
            'tie_weights': False,  # if true set LayerByLayer to False
            'filter_height': 3,
            'filter_width': 3,
            'num_filters': 64,
            'n_DnCNN_layers': 16,
            'max_n_DAMP_layers': 10,
            # Unless FLAGS.start_layer is set to this value or LayerbyLayer=false, the code will sequentially train larger and larger networks end-to-end.

            ## Training Parameters
            'start_layer': 1, # used to define where to start when resuming
            'max_Epoch_Fails': 3,  # How many training epochs to run without improvement in the validation error
            'ResumeTraining': False,  # Load weights from a network you've already trained a little
            'LayerbyLayer': True,
            'DenoiserbyDenoiser': False,  # if this is true, overrides other two
            'sigma_w_min': 25, # only used in denoiserbydenoiser training
            'sigma_w_max': 25, # only used in denoiserbydenoiser training
            'sigma_w': 1./255.,  # Noise std (LbL testing: 0, LbL training: 1./255., DbD test and train: 25./255.)
            'learning_rates': [0.001, 0.0001],  # , 0.00001]
            'EPOCHS': 50,
            'n_Train_Images': 128 * 1600,  # 128*3000
            'n_Val_Images': 10000,  # 10000
            'n_Test_Images': 50, # only for LbL
            'BATCH_SIZE':128, # 128 for training, 1 for testing
            'InitWeightsMethod': 'smaller_net', #Options are random, denoiser, smaller_net, and layer_by_layer.
            'loss_func': 'MSE',
            'init_mu': 0,
            'init_sigma': 0.1,
            'mode': sensingtype,

            'validation_patch': './Data/ValidationData_patch40.npy',
            # 'training_patch': './Data/TrainingData_patch40.npy',
            'training_patch': './Data/bigset_train_data.npy',
            # 'testing_patch': './Data/StandardTestData_256Res.npy',
            'testing_patch': './Data/bigset_test.npy',

            # if True, will use 'validation_patch' to load val data, otherwise cut from training_patch
            'use_separate_val_patch': False, # TODO LOOK AT NOTES

            # if True, creates new dataset and uses it, False: uses 'training_patch'
            'create_new_dataset': False,
            'new_data': '/storage-t1/database/bigset/train/data/*.bmp',
            'dataset_custom_name': 'bigset_train_data',

            # if True, creates new dataset and uses it, False: uses 'testing_patch'
            'create_new_testpatch': False,  # if False, ignore parameters below
            'new_test_data': '/storage-t1/database/bigset/test/*.bmp',
            'testset_custom_name': 'bigset_test1'
        },
    )