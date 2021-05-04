import ISTANet.reconstruction_methods as rms
import ISTANet.utils as utils

def main(sensing,reconstruction,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default==True:
        dataset = 'Training_Data.mat'
        input_channel = 1 # restricted to 1
        input_width = 33 # restricted to 33
        input_height = 33 # restricted to 33
        m = 272
        n = 1089 # ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
        specifics = {
            'sudo_rgb': False,
            'start_epoch': 0,
            'end_epoch': 200,
            'testing_epoch_num': 60,
            'learning_rate': 1e-4,
            'layer_num': 9,
            'group_num': 2, # check to make sure group matches folder
            'cs_ratio': 25,
            'nrtrain': 88912,
            'batch_size': 64,

            'model_dir': 'model',
            'log_dir': 'log',
            'data_dir': 'data',
            'result_dir': 'result',
            'matrix_dir': 'sampling_matrix',

            'use_universal_matrix': False, # unused in original
            'create_custom_dataset': False, # unused in original
            'custom_dataset_name': "", # unused in original
            'custom_training_data_location': '', # unused in original
            'custom_type_of_image': '', # unused in original

            'training_data_fileName': 'Training_Data',
            'training_data_type': 'mat',
            'Testing_data_location': 'Set11',
            'testing_data_isFolderImages': True,
            'testing_data_type': 'tif',
        }

        # with open('./original_specifics.json', 'r') as fp:
        #     specifics = json.load(fp)

    # Transfer parameters to specifics
    specifics['n'] = n
    specifics['m'] = m
    specifics['input_channel'] = input_channel
    specifics['input_width'] = input_width
    specifics['stage'] = stage
    dset, Training_Labels = utils.generate_dataset(input_channel,input_width,input_height, stage, specifics)
    # sensing_method=sms.sensing_method(sensing, specifics) # not used
    reconstruction_method=rms.reconstruction_method(reconstruction,specifics)
    reconstruction_method.initialize(dset, Training_Labels, specifics)
    reconstruction_method.run()

if __name__ == "__main__":
    input_channel = 1
    input_width = 64
    ratio = 2
    stage = 'training'

    # n = int(ratio/100 * m)
    n = input_width**2
    m = n // ratio

    print("Ratio: " + str(m/n)) # it is multiplied in from the right not the left
    main(
        "random",
        "ISTANetPlus", # "ISTANet", "ISTANetPlus"
        stage,
        False, # use paper's original parameters/data
        0, # dataset unused
        1, # channel
        input_width,
        input_width,
        m,
        n,
        specifics={
            'sudo_rgb': True, # if dataset is rgb, then true else false
            'start_epoch': 0,
            'end_epoch': 50,
            'testing_epoch_num': 50,
            'learning_rate': 1e-4,
            'layer_num': 9,
            'group_num': 901, # organizational purposes, 900-905
            'cs_ratio': ratio,
            'nrtrain': 60000, #224000, #88912, 867, 224000, 50000 * 3
            'batch_size': 128, # 64 for train

            # customize your directory names
            'model_dir': 'model',
            'log_dir': 'log',
            'data_dir': 'data',
            'result_dir': 'result',
            'matrix_dir': 'sampling_matrix',
            'qinit_dir': 'qinits',

            # Typically keep True
            # (set to False to use original paper's preloaded matrices which only supports imgs 33x33)
            'use_universal_matrix': True,

            # (Training only) if not creating new dataset, input file name and type.
            # Will then train on this dataset
            'training_data_fileName': 'benchmark_celebA_64x64_train', # benchmark_bigset_gray_train
            # 'Training_Data', 'bigset_train_data', special cases: 'mnist','cifar10','celeba'
            'training_data_type': 'npy',  # mat, npy

            # (Training only) set to True and define custom name, location, and type.
            # Will then create and train on this new dataset
            'create_custom_dataset': False,
            'custom_dataset_name': "benchmark_celebA_64x64_train", # what you want to call it
            'custom_training_data_location': '/storage-t1/database/cs-framework-database/celebA_64x64/train', # bigset
            'custom_type_of_image': 'jpg', # bmp, tif, celebA: jpg, mnist, cifar10: png

            # (Testing only) if testing, use these parameters to define where and what type of images testing on
            'Testing_data_location': '/storage-t1/database/cs-framework-database/celebA_64x64/test',# 'Set11'
            'testing_data_isFolderImages': True,
            'testing_data_type': 'jpg',  # bigset: bmp, celebA: jpg, mnist, cifar10: png
        }
    )
