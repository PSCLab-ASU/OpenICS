import reconstruction_methods as rms
import utils
import sensing_methods as sms
def main(sensing,reconstruction,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        CS_ratio = 25
        n_input = 272
        n_output = 1089
        batch_size = 64
        PhaseNumber = 5
        nrtrain = 88912
        learning_rate = 0.0001
        EpochNum = 300

    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage)
    sensing_method=sms.sensing_method(sensing,m,specifics)
    reconstruction_method=rms.reconstruction_method(reconstruction,specifics)
    # put result of the parameters into specifics.
    reconstruction_method.initialize(dset,sensing_method,specifics)
    reconstruction_method.run()
        

if(__name__=='__main__'):
    print('lit')