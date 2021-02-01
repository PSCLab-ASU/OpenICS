import numpy as np
lasso = {
            'pretrained-model-dir': './models/mnist-vae/',

            'input-type': "full-input",
            'num-input-images' : 300,
            'batch-size': 50,
            'noise-std': 0.1,

            'model-types': ["lasso"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.0,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':0,
            'num-random-restarts': 0,
            'lmbd': 0.1,

            'save-stats' : False,
            'print-stats' : True,
}
vae = {
            'pretrained-model-dir': './models/mnist-vae/',

            'input-type': "full-input",
            'num-input-images' : 300,
            'batch-size': 50,
            'noise-std': 0.1,

            'model-types': ["vae"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.0,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':1000,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
vaeWithReg = {
            'pretrained-model-dir': './models/mnist-vae/',

            'input-type': "full-input",
            'num-input-images' : 300,
            'batch-size': 50,
            'noise-std': 0.1,

            'model-types': ["vae"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.1,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':1000,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
vaeGenspan = {
            'pretrained-model-dir': './models/mnist-vae/',

            'input-type': "gen-span",
            'num-input-images' : 300,
            'batch-size': 50,
            'noise-std': 0.1,

            'model-types': ["vae"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.1,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':1000,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
lassoWavelet = {
            'pretrained-model-dir': './models/celebA_64_64/',

            'input-type': "full-input",
            'num-input-images' : 64,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["lasso-wavelet"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':0,
            'num-random-restarts': 0,
            'lmbd': 0.00001,

            'save-stats' : False,
            'print-stats' : True,
}
lassoDCT = {
            'pretrained-model-dir': './models/celebA_64_64/',

            'input-type': "full-input",
            'num-input-images' : 64,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["lasso-dct"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':0,
            'num-random-restarts': 0,
            'lmbd': 0.1,

            'save-stats' : False,
            'print-stats' : True,
}
dcganWithReg = {
            'pretrained-model-dir': './models/celebA_64_64/',

            'input-type': "full-input",
            'num-input-images' : 64,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["dcgan"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.001,
            'dloss1_weight':1.0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':500,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
dcgan = {
            'pretrained-model-dir': './models/celebA_64_64/',

            'input-type': "full-input",
            'num-input-images' : 64,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["dcgan"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.00,
            'dloss1_weight':0.0,
            'dloss2_weight':0.0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':500,
            'num-random-restarts': 2,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,


            'validation_patch': './data/ValidationData_patch40.npy',
            'training_patch': './data/benchmark_mnist_train.npy',
            'testing_patch': './data/celebA_64x64_test.npy',

            # if True, will use 'validation_patch' to load val data, otherwise cut from training_patch
            'use_separate_val_patch': False, # TODO LOOK AT NOTES

            # if True, creates new dataset and uses it, False: uses 'training_patch'
            'create_new_dataset': True,
            'new_data': '/storage-t1/database/cs-framework-database/celebA_64x64/train',
            'dataset_custom_name': 'celebA_64x64_test_train',
            'custom_type_of_image': 'jpg',  # bmp, tif, jpg, png

            # if True, creates new dataset and uses it, False: uses 'testing_patch'
            'create_new_testpatch': False,  # if False, ignore parameters below
            'new_test_data': '/storage-t1/database/cs-framework-database/celebA_64x64/test',
            'testset_custom_name': 'celebA_test',
            'testing_data_type': 'jpg'  # bmp, tif, celebA: jpg, mnist, cifar10: png
}
dcganCustom = {
            'pretrained-model-dir': './models/celebA_64_64_pretrained/',
            'test_data': './data/celebAtest/*.jpg',
            'input-type': "full-input",
            'num-input-images' : 1,
            'batch-size': 1,
            'noise-std': 0.01,

            'model-types': ["dcgan"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.00,
            'dloss1_weight':0.0,
            'dloss2_weight':0.0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':5,
            'num-random-restarts': 2,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,


}
dcganWithRegGenspan = {
            'pretrained-model-dir': './models/celebA_64_64/',

            'input-type': "gen-span",
            'num-input-images' : 64,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["dcgan"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.001,
            'dloss1_weight':0.0,
            'dloss2_weight':0.0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':500,
            'num-random-restarts': 1,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}

custom = {
            'pretrained-model-dir': './mnist_vae/models/mnist-vae/',

            'input-type': "full-input",
            'num-input-images' : 100,
            'batch-size': 50,
            'noise-std': 0.1,

            'model-types': ["vae"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.0,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':100,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
vaeTrain = {
    'num-samples': 60000,
    'train_data': './data/mnist_benchmark/train/*.png',
    'learning-rate': 0.001,
    'batch-size': 100,
    'input-size': 32,
    'training-epochs': 200,
    'summary-epochs': 1,
    'ckpt_epoch': 5,
}
dcganTrain = {
    'epoch': 300,
    'learning_rate': 0.0002,
    'beta1':0.5,
    'train_size': np.inf,
    'batch_size':64,
    'input_height':32,
    'input_width':None,
    'output_height':32,
    'output_width':None,
    'dataset':"cifar10_gray/train",
    'input_fname_pattern':"*.png",
    'data_dir':'./data',
    'out_dir':'./models',
    'out_name':'cifar_10_grayyyy',
    'checkpoint_dir': 'checkpoint',
    'sample_dir':'samples',
    'train':True,
    'crop':True,
    'visualize':False,
    'export':False,
    'freeze':False,
    'max_to_keep':None,
    'sample_freq':200,
    'ckpt_freq':200,
    'z_dim': 100,
    'z_dist':'uniform_signed',
    'G_img_sum': False
}
dcganWithRegCustom = {
            'pretrained-model-dir': './models/celebA_64_64/',
            'test_data': './data/celebAtest/*.jpg',
            'input-type': "full-input",
            'num-input-images' : 867,
            'batch-size': 64,
            'noise-std': 0.01,

            'model-types': ["dcgan"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.001,
            'dloss1_weight':1.0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.1,
            'momentum':0.9,
            'max-update-iter':500,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}
vaeWithRegCustom = {
            'pretrained-model-dir': './models/mnist-vae/',
            'test_data': './data/mnist_benchmark/test/*.png',
            'input-type': "full-input",
            'num-input-images' : 10000,
            'batch-size': 100,
            'noise-std': 0.1,

            'model-types': ["vae"],
            'mloss1_weight':0.0,
            'mloss2_weight':1.0,
            'zprior_weight':0.1,
            'dloss1_weight':0,
            'dloss2_weight':0,

            
            'optimizer-type': "adam",
            'learning-rate': 0.01,
            'momentum':0.9,
            'max-update-iter':1000,
            'num-random-restarts': 10,
            'lmbd': 0,

            'save-stats' : False,
            'print-stats' : True,
}