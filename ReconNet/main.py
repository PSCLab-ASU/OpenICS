import reconstruction_methods as rms
import utils
import sensing_method as sms


def main(sensing, reconstruction, stage, default, dataset, input_channel, input_width, input_height, m, n, specifics):
    if default == "True":
    # set all the parameters with the default values.

    else:
        dset = utils.generate_dataset(dataset, input_channel, input_width, input_height, stage)
        sensing_method = sms.sensing_method(sensing, m, specifics)
        reconstruction_method = rms.reconstruction_method(reconstruction, specifics)
        # put result of the parameters into specifics.
        reconstruction_method.initialize(dset, sensing_method, specifics)
        reconstruction_method.run("train")
        reconstruction_method.run("test")


