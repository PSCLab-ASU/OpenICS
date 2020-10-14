import reconstruction_methods as rms
import utils
import sensing_methods as sms

def main(sensing,reconstruction,stage,default,dataset,input_channel,input_width,input_height,m,n,specifics):
    if default=="True":
        dataset = 'Training_Data.mat'
        input_channel = 0 # computed automatically
        input_width = 0 # computed automatically
        input_height = 0 # computed automatically
        m = 1089
        n = 272
        specifics = {
            'stage': 'training',
            'start_epoch': 0,
            'end_epoch': 200,
            'testing_epoch_num': 60,
            'learning_rate': 1e-4,
            'layer_num': 9,
            'group_num': 1,
            'cs_ratio': 25,
            'gpu_list': 0,
            'n_input': n,
            'n_output':m,
            'nrtrain': 88912,
            'batch_size': 64,
            'model_dir': 'model',
            'log_dir': 'log',
            'data_dir': 'data',
            'result_dir': 'result',
            'matrix_dir': 'sampling_matrix',
            'Training_data_Name': dataset,
            'test_name': 'Set11'
        }


    dset = utils.generate_dataset(stage, specifics)
    # sensing_method=sms.sensing_method(sensing, specifics) # not used
    reconstruction_method=rms.reconstruction_method(reconstruction,specifics)
    reconstruction_method.initialize(dset, specifics)
    reconstruction_method.run()
        
if __name__ == "__main__":
    m = 1089
    n = 272 # ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
    dataset = 'Training_Data.mat'
    stage = 'training'
    main(
        "random",
        "ISTANetPlus",
        stage,
        False,
        dataset,
        0,
        0,
        0,
        m,
        n,
        specifics={
            'stage': stage,
            'start_epoch': 0,
            'end_epoch': 200,
            'testing_epoch_num': 200,
            'learning_rate': 1e-4,
            'layer_num': 9,
            'group_num': 1,
            'cs_ratio': 30,
            'gpu_list': 0,
            'n_input': n,
            'n_output': m,
            'nrtrain': 88912,
            'batch_size': 64, # 1 for test, 64 for train
            'model_dir': 'model',
            'log_dir': 'log',
            'data_dir': 'data',
            'result_dir': 'result',
            'matrix_dir': 'sampling_matrix',
            'Training_data_Name': dataset,
            'test_name': 'Set11'
        }
    )
        