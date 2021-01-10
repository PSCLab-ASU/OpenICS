import reconstruction_methods as rms
import utils
import sensing_methods as sms
import numpy as np
import presets
def main(reconstruction,sensing,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        sensing = "gaussian"
        reconstruction = "csgm"
        stage = stage
        dataset = dataset # '.\exampleLocation' # location of dataset
        dataset = "mnist"

        input_channel = 1 # n_channels
        input_width = 28 # width
        input_height = 28 # height
        n = 28*28*1
        m = 50 # 10 25 50 100 200 300 400 500 750
        specifics = presets.dcganTrain


    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage=stage)
    sensing_method= sms.sensing_method(sensing,n,m,input_width,input_channel)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing_method, dataset,sensing,specifics)

    reconstruction_method.initialize(dset,sensing_method,stage,specifics)
    reconstruction_method.run(stage)
        

if __name__ == "__main__":
    specifics = presets.dcganWithRegCustom
    #print(specifics)
    main(
        reconstruction= "csgm",
        sensing = "gaussian", #Can be either sensing_matrix or a neural network (NN_mnist or NN_celeba)
        stage = "testing",
        default = "False",
        dataset = "celebA",
        input_channel = 1,
        input_width = 32,
        input_height = 32,
        n = 32*32*1,
        m = 512, # 10 25 50 100 200 300 400 500 750
        specifics = specifics
    )