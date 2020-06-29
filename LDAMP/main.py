import reconstruction_methods as rms
import utils
import sensing_methods as sms
import numpy as np
import torch


def main(sensing, reconstruction, stage, default, dataset, input_channel, input_width, input_height, m, n, specifics):
    if default == "True":
        sensing = "Gaussian"
        reconstruction = "LearnedDAMP"
        stage = stage
        dataset = dataset  # './TrainingData/ValidationData_patch'
        input_channel = 1
        input_width = 64 # original paper was 40
        input_height = 64 # original paper was 40
        n = input_channel * input_width * input_height
        sampling_rate = .2
        m = int(np.round(sampling_rate * n))
        if (stage == "training"):
            specifics = {
                # "alg": "DAMP", # only worried about LDAMP
                "tie_weights": False,  # if true, only train last layer of network
                "filter_height": 3,
                "filter_width": 3,
                "num_filters": 64,
                "n_DnCNN_layers": 16,
                "max_n_DAMP_layers": 10,
                "start_layer": 1,
                "max_Epoch_Fails": 3,  # How many training epochs to run without improvement in the validation error
                "ResumeTraining": False,  # Load weights from a network you've already trained a little
                "LayerbyLayer": False,  # default is false which means train end-to-end
                "learning_rate": 0.0001,  # [0.001, 0.0001], 0.00001],
                "EPOCHS": 50,
                "n_Train_Images": 128 * 1600,  # 128*3000
                "n_Val_Images": 10000,  # 10000#Must be less than 21504
                "BATCH_SIZE": 1, # 128,
                "InitWeightsMethod": "denoiser",
                "BATCH_SIZE_LBLFALSE": 16,
                "loss_func": "MSE",
                "sampling_rate": .2,  # The sampling rate that was used for training
                "sigma_w": 1. / 255.,  # Noise std
                "init_mu": 0,
                "init_sigma": 0.1,
                "m": m,
                "n": n,
                "input_channel": input_channel,
                "input_width": input_width,
                "input_height": input_height
            }
        else:
            specifics = {
                # "alg": "DAMP", # only worried about LDAMP
                "tie_weights": False,
                "filter_height": 3,
                "filter_width": 3,
                "num_filters": 64,
                "n_DnCNN_layers": 16,
                "max_n_DAMP_layers": 10,
                "BATCH_SIZE": 1,
                # Using a batch size larger than 1 will hurt the denoiser by denoiser trained network because it will use an average noise level, rather than a noise level specific to each image
                "n_Test_Images": 5,
                "sampling_rate_test": .2,  # The sampling rate used for testing
                "sampling_rate_train": .2,  # The sampling rate that was used for training
                "sigma_w": 0.,
                "init_mu": 0,
                "init_sigma": 0.1,
                "m": m,
                "n": n,
                "input_channel": input_channel,
                "input_width": input_width,
                "input_height": input_height
            }
            if (not (stage == 'testing')):
                print(
                    'stage "' + stage + '" cannot be recognized. Stage must be either testing or training. Default will be used (testing)')

    dset = utils.generate_dataset(dataset, input_channel, input_width, input_height, stage, specifics=specifics)
    sensing_method = sms.sensing_method(sensing, specifics, m, n=input_channel * input_width * input_height)
    reconstruction_method = rms.reconstruction_method(reconstruction, specifics)
    # put result of the parameters into specifics.
    reconstruction_method.initialize(dset, sensing_method, specifics)
    reconstruction_method.run(stage)


if __name__ == "__main__":
    main(
        "Gaussian",
        "LearnedDAMP",
        "training",
        "True",
        "/storage-t1/temp/bsd500patch/data",
        # "/home/mkweste1/LDAMP final/Data/TrainingData/StandardTestData_256Res.npy",
        1,
        40,
        40,
        40 * 40 * 1,
        int(np.round(.2 * 40 * 40 * 1)),
        {
            "tie_weights": False,
            "filter_height": 3,
            "filter_width": 3,
            "num_filters": 64,
            "n_DnCNN_layers": 16,
            "max_n_DAMP_layers": 10,
            "BATCH_SIZE": 1,
            # Using a batch size larger than 1 will hurt the denoiser by denoiser trained network because it will use an average noise level, rather than a noise level specific to each image
            "n_Test_Images": 5,
            "sampling_rate_test": .2,  # The sampling rate used for testing
            "sampling_rate_train": .2,  # The sampling rate that was used for training
            "sigma_w": 0.,
            "init_mu": 0,
            "init_sigma": 0.1
        },
    )
