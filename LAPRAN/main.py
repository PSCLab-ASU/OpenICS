import reconstruction_methods as rms
import utils
import sensing_methods as sms
import argparse

trainSpecifics = {
            "model": 'adaptiveCS_resnet_wy_ifusion_ufirst',
            "dataset": "cifar10",
            "batch_size": 128,
            "test_batch_size": 1000,
            "epochs": 100,
            "lr": 2e-4,
            "momentum": 0.5,
            "cuda":True,
            "ngpu": 1,
            "seed": 1,
            "log-interval": 100,
            "layers-gan": 3,
            "gpu": 0,
            "outf": './results',
            "w-loss": 0.01,
            "stage": 4, #stage under training
            "transfer": False,
            "cr":10
            }
            
testSpecifics = {
            "model": 'adaptiveCS_resnet_wy_ifusion_ufirst',
            "dataset": "cifar10",
            "batch_size": 1,
            "test_batch_size": 1000,
            "epochs": 100,
            "lr": 2e-4,
            "momentum": 0.5,
            "cuda":True,
            "ngpu": 1,
            "seed": 1,
            "log-interval": 100,
            "layers-gan": 3,
            "gpu": 0,
            "outf": './results',
            "w-loss": 0.01,
            "stage": 1, #stage under training
            "transfer": False,
            "cr":10
            }
def main(reconstruction, sensing = "random_sensing",stage = "testing",default = "True",dataset = "cifar10",input_channel=3,input_width=64,input_height=64,n = 64**2, m = 64**2//10 ,specifics = None):
    if default=="True":
        sensing = "random_sensing"
        reconstruction = "LAPRAN"
        stage = "testing"
        dataset = "cifar10"
        input_channel = 3
        input_width = 64
        input_height = 64
        n = 64**2
        m = n//10
        specifics = testSpecifics
    
    dset=utils.generate_dataset(dataset,input_channel,input_width,input_height,stage, specifics)
    sensing_method=sms.sensing_method(sensing,n,m,input_width,input_channel)
    reconstruction_method=rms.reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n, specifics)
    reconstruction_method.initialize(dset,sensing_method,stage, specifics)
    reconstruction_method.run(stage, specifics)
        
if __name__ == '__main__':
    main("LAPRAN")        
