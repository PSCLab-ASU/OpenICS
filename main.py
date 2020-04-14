import os
import reconstruction_methods as rms
import utils
import sensing_methods as sms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(
    sensing,
    reconstruction,
    stage,
    default,
    dataset,
    input_channel,
    input_width,
    input_height,
    m,
    n,
    specifics,
):
    if default == "True" and reconstruction == "ISTANet":
        sensing = "Gaussian"
        # TODO: download training_data.mat onto the server
        dataset = "Set11"
        input_channel = 1
        input_width = 256
        input_height = 256
        # ISTANet describes this value in terms of "CS Ratio".
        # They test for a bunch of CS Ratios but I'll use 0.25 as the default.
        # Other CS Ratios in the paper:
        # 50, 40, 30, 25, 10, 4, 1
        n = input_height * input_width
        specifics = {
            "batch_size": 64,
            "layer_num": 9,
            "learning_rate": 1e-4,
            "start_epoch": 0,
            "end_epoch": 200,
            "cs_ratio": 25,
            "epoch_num": 200,
        }
        m = n * specifics["cs_ratio"] / 100

    if stage == "training":
        dset = utils.generate_dataset(
            dataset, "mat", input_channel, input_width, input_height, stage
        )

        sensing_method = sms.sensing_method(sensing, dset, m, specifics)
    else:
        # TODO: make utils method for reading tiffs.
        sensing_method = None
    reconstruction_method = rms.reconstruction_method(reconstruction, specifics)
    # put result of the parameters into specifics.
    reconstruction_method.initialize(sensing_method)
    reconstruction_method.run(stage)


if __name__ == "__main__":
    main(
        "Gaussian",
        "ISTANet",
        "testing",
        "True",
        "models/Training_Data.mat",
        1,
        0,
        0,
        0,
        0,
        {
            "batch_size": 64,
            "layer_num": 9,
            "learning_rate": 1e-4,
            "start_epoch": 0,
            "end_epoch": 200,
            "cs_ratio": 25,
        },
    )
