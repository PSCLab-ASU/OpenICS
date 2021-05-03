import ReconNet.reconstruction_methods as rms
import ReconNet.utils as utils
import ReconNet.sensing_method as sms

custom_specifics = {
    'test-root': '/storage-t1/database/cs-framework-database/mnist/test',
    'train-root': '/storage-t1/database/cs-framework-database/mnist/train',
    'model-root': './models',
    'logs-root': './logs',
    
    'epochs': 100,
    'lr': 1e-4,
    'betas': (0.5, 0.999),
    'batch-size': 1,
    
    #'specific-model-path': './models/reconnet/mnist/cr2/best_model.rnet',
    #'sensing-path': './models/reconnet/mnist/cr2/rand.rnet.sensing',
    #'resume-training': False,
    
    'save-interval': 5,
    #'max-images': 16,
    'validation-images': 1 / 5,
    'validation-split-seed': 2147483647,
    'workers': 2,
    'device': 'cuda:0'
}

def main(sensing, reconstruction, stage, default, dataset, input_channel, input_width, input_height, m, n, specifics):
    if str(default) == 'True':
        # set all the parameters with the default values.
        sensing = 'random'
        reconstruction = 'reconnet'
        stage = 'testing'
        dataset = 'mnist'
        input_channel = 1
        input_width = 32
        input_height = 32
        cr = 2
        
        m = input_width * input_height // cr
        n = input_channel * input_width * input_height
        
        specifics = {
            'epochs': 50,
            'batch-size': 128,
            'lr': 2e-5,
            'betas': (0.5, 0.999),
            'test-root': '/storage-t1/database/cs-framework-database/mnist/test',
            'train-root': '/storage-t1/database/cs-framework-database/mnist/train',
            'model-root': './models',
            'logs-root': './logs',
        }

    dset = utils.generate_dataset(dataset, input_channel, input_width, input_height, stage, specifics)
    sensing_method = sms.sensing_method(sensing, n, m, input_width, input_height, input_channel)
    reconstruction_method = rms.reconstruction_method(reconstruction, input_channel, input_width, input_height, m, n, specifics)
    
    reconstruction_method.initialize(dset, sensing_method, stage)
    reconstruction_method.run(stage)

if __name__ == "__main__":
    input_channel, input_width, input_height = 1, 32, 32
    ratio = 2
    stage = 'testing'
    dataset = 'mnist'
    
    # NOTE: m is per-channel, so for input size of w,h,c and compression ratio r, m = w * h // r
    # n, on the other hand, is the total size, so for input size of w,h,c, n = w * h * c

    main(
        sensing='random',
        reconstruction='ReconNet',
        stage=stage,
        default='False',
        dataset=dataset,
        input_channel=input_channel,
        input_width=input_width,
        input_height=input_height,
        m=input_width * input_height // ratio,
        n=input_width*input_height*input_channel,
        specifics=custom_specifics
    )
