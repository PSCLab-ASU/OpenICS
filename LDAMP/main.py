import reconstruction_methods as rms
import utils
import sensing_methods as sms
import numpy as np
import torch


def main(sensing, reconstruction, stage, default, dataset, input_channel, input_width, input_height, m, n, specifics):
    if default:
        sensing = "Gaussian"
        reconstruction = "LearnedDAMP"
        stage = stage
        dataset = dataset
        input_channel = 1
        input_width = 40
        input_height = 40
        sampling_rate = .2
        m = int(np.round(sampling_rate * input_channel * input_width * input_height))
        n = input_channel * input_width * input_height

        if (stage == "training"):
            specifics = {
                "filter_height": 3,# kernel size
                "filter_width": 3,# kernel size
                "num_filters": 64, # channel size, in dncnn
                "n_DnCNN_layers": 16,
                "max_n_DAMP_layers": 1,
                "learning_rate": 0.00001,
                "EPOCHS": 50, # 50,
                "n_train_images": 1103872, #1103872
                "n_val_images": 10000,
                "BATCH_SIZE": 128,
                "loss_func": "MSE",
                "sampling_rate": sampling_rate,  # The sampling rate that was used for training
                "sigma_w": 1. / 255.,  # Noise std
                "m": m,
                "n": n,
                "input_channel": input_channel,
                "input_width": input_width,
                "input_height": input_height,
                "resume": False,
                "resuming_network": '',
                "previously_trained": 0,
                "fileName": "LDAMPFirst"
            }
        else:
            specifics = {
                "filter_height": 3,
                "filter_width": 3,
                "num_filters": 64,
                "n_DnCNN_layers": 16,
                "max_n_DAMP_layers": 4,# 10,
                "BATCH_SIZE": 1,
                "n_val_images": 5,
                "sampling_rate_test": .2,  # The sampling rate used for testing
                "sampling_rate_train": .2,  # The sampling rate that was used for training
                "sigma_w": 0.,
                "m": m,
                "n": n,
                "input_channel": input_channel,
                "input_width": input_width,
                "input_height": input_height,
                "resume": False,
                "load_network": '',
                "previously_trained": 0,
                "fileName": "LDAMPFirst"
            }

    dset = utils.generate_dataset(dataset, input_channel, input_width, input_height, stage, specifics=specifics)
    sensing_method = sms.sensing_method(sensing, specifics, m, n=n)
    reconstruction_method = rms.reconstruction_method(reconstruction, specifics)
    reconstruction_method.initialize(dset, sensing_method, specifics)
    reconstruction_method.run(stage)


if __name__ == "__main__":
    main(
        "Gaussian", # sensing type
        "LearnedDAMP", # model type
        "training", # stage
        False, # default
        '/home/user/mkweste1/LDAMP final/Data/TrainingData/TrainingData_patch40.npy', # dataset
        # "/home/user/mkweste1/LDAMP final/Data/TrainingData/StandardTestData_256Res.npy",
        1, # input channels
        40, # input width
        40, # input height
        320,  # m 320
        40 * 40 * 1, # n 40x40x1
        {
            "filter_height": 3,  # kernel size
            "filter_width": 3,  # kernel size
            "num_filters": 64,  # channel size
            "n_DnCNN_layers": 16, #tf code: 16 paper: 20
            "max_n_DAMP_layers": 10, #tf_code: 10 paper: 10
            "learning_rate": 0.0001,
            "EPOCHS": 50,
            "n_train_images": 128*1600,#1103872,
            "n_val_images": 1000,
            "BATCH_SIZE": 64,  # 18,
            "loss_func": "MSE",
            "sampling_rate": 320,
            "sigma_w": 1. / 255.,  # Noise std
            "m": 320,
            "n": 40 * 40 * 1,
            "input_channel": 1,
            "input_width": 40,
            "input_height": 40,
            "load_network": './LDAMP saved models/quickSaveLDAMP10Layer7372800', #resumeing from 36 EPOCHS with newly generated sensing matrices now
            "resume": False,
            "fileName": "LDAMP"
        },
    )
