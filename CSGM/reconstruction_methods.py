import os
from argparse import ArgumentParser
from argparse import Namespace
import numpy as np
import CSGM.utils as utils
import tensorflow.compat.v1 as tf
import CSGM.dcgan.dcgan_utils as dcgan_utils
from glob import glob
import time
import scipy.misc
def reconstruction_method(reconstruction,input_channel, input_width, input_height, m, n,sensing,datasetname, sensingname, specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if (reconstruction == 'csgm' or reconstruction == 'CSGM'):
        CSGM = csgm(input_channel, input_width, input_height, m, n,sensing, datasetname,sensingname,specifics)
        return CSGM

class csgm():
    def __init__(self, input_channel, input_width, input_height, m, n, sensing,datasetname,sensingname, specifics):
        self.channels = input_channel
        self.input_width = input_width
        self.input_height = input_height
        self.m = m 
        self.n = n
        self.datasetname = datasetname
        self.sensingname = sensingname
        self.specifics = specifics

        self.hparams = None
    def initialize(self,dset,sensing_method,stage,specifics): 
        self.sensing_method = sensing_method
        self.model_input = dset

        if (stage =="testing"):
            data_path = specifics['test_data']
            self.data = glob(data_path)
            if len(self.data) == 0:
                raise Exception("[!] No data found in '" + data_path + "'")
            np.random.shuffle(self.data)
            imreadImg = scipy.misc.imread(self.data[0]).astype(np.float)
            if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
                self.c_dim = scipy.misc.imread(self.data[0]).astype(np.float).shape[-1]    
            else:
                self.c_dim = 1
            self.grayscale = (self.c_dim == 1)
            print("GRAYSCALE:", self.grayscale)
            HPARAMS = Namespace(
                input_size = self.input_width,
                input_channels = self.channels,
                pretrained_model_dir = specifics["pretrained-model-dir"],
                grayscale = self.grayscale,
                dataset=self.datasetname,
                input_type =specifics["input-type"],
                input_path_pattern= specifics["test_data"],
                num_input_images=specifics["num-input-images"],
                batch_size=specifics["batch-size"],   
                measurement_type=self.sensingname,
                noise_std=specifics["noise-std"],

                num_measurements=self.m,
                inpaint_size=1,
                superres_factor=2,

                
                model_types= specifics["model-types"],
                mloss1_weight=specifics["mloss1_weight"],
                mloss2_weight=specifics["mloss2_weight"],
                zprior_weight=specifics["zprior_weight"],
                dloss1_weight=specifics["dloss1_weight"],
                dloss2_weight=specifics["dloss2_weight"],

                
                optimizer_type=specifics["optimizer-type"],
                learning_rate=specifics["learning-rate"],
                momentum=specifics["momentum"],
                max_update_iter=specifics["max-update-iter"],
                num_random_restarts=specifics["num-random-restarts"],
                decay_lr=False,
                lmbd=specifics["lmbd"],
                lasso_solver='sklearn',
                sparsity=0,

                
                not_lazy =False,
                save_images=True,
                save_stats=specifics["save-stats"],
                print_stats=specifics["print-stats"],
                checkpoint_iter=1,
                image_matrix=0,
                gif = False,
                gif_iter=1,
                gif_dir=''
            )
            if HPARAMS.dataset == 'mnist':
                HPARAMS.image_shape = (self.input_width, self.input_height, self.channels)
                from CSGM.mnist.mnist_utils import view_image, save_image
                self.view_image = view_image
                self.save_image = save_image
            elif HPARAMS.dataset == 'celebA':
                HPARAMS.image_shape = (self.input_width, self.input_height, self.channels)
                from CSGM.celebA.celebA_utils import view_image, save_image
                self.view_image = view_image
                self.save_image = save_image
            else:
                raise NotImplementedError

            self.hparams = HPARAMS

    
    def run(self, stage):
        if (stage =="testing"):
            hparams = self.hparams
            model_input = self.model_input
            view_image = self.view_image
            save_image = self.save_image
            # Set up some stuff accoring to hparams
            hparams.n_input = np.prod(hparams.image_shape)
            utils.set_num_measurements(hparams)
            utils.print_hparams(hparams)

            # get inputs
            xs_dict = model_input(hparams)
            estimators = utils.get_estimators(hparams)
            utils.setup_checkpointing(hparams)
            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
            PSNRs = {}
            SSIMs = {}
            reconst_times = []
            for model_type in hparams.model_types:
                PSNRs[model_type] = {}
                SSIMs[model_type] = {}
            recons_times = []

            x_hats_dict = {model_type : {} for model_type in hparams.model_types}
            x_batch_dict = {}
            for key, x in xs_dict.items():

                if not hparams.not_lazy:
                    # If lazy, first check if the image has already been
                    # saved before by *all* estimators. If yes, then skip this image.
                    save_paths = utils.get_save_paths(hparams, key)
                    is_saved = all([os.path.isfile(save_path) for save_path in list(save_paths.values())])
                    #if is_saved:
                    #   continue

                x_batch_dict[key] = x
                if len(x_batch_dict) < hparams.batch_size:
                    continue

                # Reshape input
                x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()]
                x_batch = np.concatenate(x_batch_list)

                # Construct noise and measurements
                A = self.sensing_method()
                noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
                if hparams.measurement_type == 'project':
                    y_batch = x_batch + noise_batch
                else:
                    y_batch = np.matmul(x_batch, A) + noise_batch

                # Construct estimates using each estimator

                for model_type in hparams.model_types:
                    estimator = estimators[model_type]
                    benchmarkStartTime = time.time()
                    x_hat_batch = estimator(A, y_batch, hparams)
                    benchmarkEndTime = time.time()
                    reconst_time = (benchmarkEndTime-benchmarkStartTime)/hparams.batch_size
                    reconst_times.append(reconst_time)
                    for i, key in enumerate(x_batch_dict.keys()):
                        x = xs_dict[key]
                        y = y_batch[i]
                        x_hat = x_hat_batch[i]

                        # Save the estimate
                        x_hats_dict[model_type][key] = x_hat

                        # Compute and store measurement and l2 loss
                        measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y)
                        l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                        if (hparams.dataset=="mnist"):
                            PSNRs[model_type][key] = utils.compute_average_psnr(x.reshape(1,hparams.input_channels,hparams.input_size,hparams.input_size)*2-1,x_hat.reshape(1,hparams.input_channels,hparams.input_size,hparams.input_size)*2-1)
                            SSIMs[model_type][key] = utils.compute_average_SSIM(x.reshape(1,hparams.input_size,hparams.input_size,hparams.input_channels)*2-1,x_hat.reshape(1,hparams.input_size,hparams.input_size,hparams.input_channels)*2-1)
                        else:
                            PSNRs[model_type][key] = utils.compute_average_psnr(x.reshape(1,hparams.input_channels,hparams.input_size,hparams.input_size),x_hat.reshape(1,hparams.input_channels,hparams.input_size,hparams.input_size))
                            SSIMs[model_type][key] = utils.compute_average_SSIM(x.reshape(1,hparams.input_size,hparams.input_size,hparams.input_channels),x_hat.reshape(1,hparams.input_size,hparams.input_size,hparams.input_channels))

                print(("Processed upto image {0} / {1}".format(key+1, len(xs_dict))))

                # Checkpointing
                if (hparams.save_images) and ((key+1) % hparams.checkpoint_iter == 0):
                    utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                    x_hats_dict = {model_type : {} for model_type in hparams.model_types}
                    print('\nProcessed and saved first ', key+1, 'images\n')
                x_batch_dict = {}

            # Final checkpoint

            if hparams.save_images:
                utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))

            if hparams.print_stats:
                for model_type in hparams.model_types:
                    print(model_type)
                    mean_m_loss = np.mean(list(measurement_losses[model_type].values()))
                    mean_l2_loss = np.mean(list(l2_losses[model_type].values()))
                    mean_PSNR = np.mean(list(PSNRs[model_type].values()))
                    mean_SSIM = np.mean(list(SSIMs[model_type].values()))
                    print('number of measurements = ' + str(self.m))
                    print('mean measurement loss = {0}'.format(mean_m_loss))
                    print('mean l2 loss = {0}'.format(mean_l2_loss))
                    print('mean PSNR = {0}'.format(mean_PSNR))
                    print('mean SSIM = {0}'.format(mean_SSIM))
                    print('mean reconstruction time = {0}'.format(sum(reconst_times)/len(reconst_times)))
                    

            if hparams.image_matrix > 0:
                utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

            # Warn the user that some things were not processsed
            if len(x_batch_dict) > 0:
                print('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
                print('Consider rerunning lazily with a smaller batch size.')
        elif(stage =="training"):
            if (self.datasetname =="mnist"):
                import CSGM.mnist_vae.main as vae
                import CSGM.mnist_vae.model_def as vae_model_def
                HPARAMS = vae_model_def.Hparams()

                HPARAMS.num_samples = self.specifics["num-samples"]
                HPARAMS.learning_rate = self.specifics["learning-rate"]
                HPARAMS.batch_size = self.specifics["batch-size"]
                HPARAMS.training_epochs = self.specifics["training-epochs"]
                HPARAMS.summary_epoch = self.specifics["summary-epochs"]
                HPARAMS.ckpt_epoch = self.specifics["ckpt_epoch"]
                HPARAMS.train_data = self.specifics["train_data"]
                HPARAMS.input_size=self.specifics["input-size"]
                HPARAMS.ckpt_dir = './models/mnist-vae/'
                HPARAMS.sample_dir = './samples/mnist-vae/'
                HPARAMS.n_input = HPARAMS.input_size*HPARAMS.input_size
                vae.main(HPARAMS)
            if (self.datasetname =="celebA"):
                import CSGM.dcgan.DCGANtensorflow.main as dcgan
                flags = tf.app.flags
                flags.DEFINE_integer("epoch", self.specifics['epoch'], "Epoch to train [25]")
                flags.DEFINE_float("learning_rate", self.specifics['learning_rate'], "Learning rate of for adam [0.0002]")
                flags.DEFINE_float("beta1", self.specifics['beta1'], "Momentum term of adam [0.5]")
                flags.DEFINE_float("train_size", self.specifics['train_size'], "The size of train images [np.inf]")
                flags.DEFINE_integer("batch_size", self.specifics['batch_size'], "The size of batch images [64]")
                flags.DEFINE_integer("input_height", self.specifics['input_height'], "The size of image to use (will be center cropped). [108]")
                flags.DEFINE_integer("input_width", self.specifics['input_width'], "The size of image to use (will be center cropped). If None, same value as input_height [None]")
                flags.DEFINE_integer("output_height", self.specifics['output_height'], "The size of the output images to produce [64]")
                flags.DEFINE_integer("output_width", self.specifics['output_width'], "The size of the output images to produce. If None, same value as output_height [None]")
                flags.DEFINE_string("dataset", self.specifics['dataset'], "The name of dataset [celebA, mnist, lsun]")
                flags.DEFINE_string("input_fname_pattern", self.specifics['input_fname_pattern'], "Glob pattern of filename of input images [*]")
                flags.DEFINE_string("data_dir", self.specifics['data_dir'], "path to datasets [e.g. $HOME/data]")
                flags.DEFINE_string("out_dir", self.specifics['out_dir'], "Root directory for outputs [e.g. $HOME/out]")
                flags.DEFINE_string("out_name", self.specifics['out_name'], "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
                flags.DEFINE_string("checkpoint_dir", self.specifics['checkpoint_dir'], "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
                flags.DEFINE_string("sample_dir", self.specifics['sample_dir'], "Folder (under out_root_dir/out_name) to save samples [samples]")
                flags.DEFINE_boolean("train", self.specifics['train'], "True for training, False for testing [False]")
                flags.DEFINE_boolean("crop", self.specifics['crop'], "True for training, False for testing [False]")
                flags.DEFINE_boolean("visualize", self.specifics['visualize'], "True for visualizing, False for nothing [False]")
                flags.DEFINE_boolean("export", self.specifics['export'], "True for exporting with new batch size")
                flags.DEFINE_boolean("freeze", self.specifics['freeze'], "True for exporting with new batch size")
                flags.DEFINE_integer("max_to_keep", self.specifics['max_to_keep'], "maximum number of checkpoints to keep")
                flags.DEFINE_integer("sample_freq", self.specifics['sample_freq'], "sample every this many iterations")
                flags.DEFINE_integer("ckpt_freq", self.specifics['ckpt_freq'], "save checkpoint every this many iterations")
                flags.DEFINE_integer("z_dim", self.specifics['z_dim'], "dimensions of z")
                flags.DEFINE_string("z_dist", self.specifics['z_dist'], "'normal01' or 'uniform_unsigned' or uniform_signed")
                flags.DEFINE_boolean("G_img_sum", self.specifics['G_img_sum'], "Save generator image summaries in log")
                #flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
                
                dcgan.main(flags)

      



