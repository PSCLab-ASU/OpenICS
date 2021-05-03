import CSGM.reconstruction_methods as rms
import CSGM.utils as utils
import CSGM.sensing_methods as sms
import numpy as np
import CSGM.presets as presets
def main(reconstruction,sensing,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        sensing = "gaussian"
        reconstruction = "csgm"
        stage = "training"
        dataset = dataset
        dataset = "mnist"

        input_channel = 1 # n_channels
        input_width = 32 # width
        input_height = 32 # height
        n = 32*32*1
        m = 32
        specifics = presets.vaeTrain


    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage=stage)
    sensing_method= sms.sensing_method(sensing,n,m,input_width,input_channel)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing_method, dataset,sensing,specifics)

    reconstruction_method.initialize(dset,sensing_method,stage,specifics)
    reconstruction_method.run(stage)
        

if __name__ == "__main__":
    specifics = presets.dcganWithRegCustom #see more in presets.py
    #print(specifics)
    main(
        reconstruction= "csgm",
        sensing = "gaussian",
        stage = "testing",
        default = "False",
        dataset = "celebA",
        input_channel = 1,
        input_width = 32,
        input_height = 32,
        n = 32*32*1, #width*height*channels
        m = 512,
        specifics = specifics
    )