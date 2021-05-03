#TODO tensorflow version 2.X migration code changed the import tensorflow as tf line to two lines as seen below
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import random
import LDAMP.sensing_methods as sensing_methods
import LDAMP.utils as utils
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
def reconstruction_method(dset,sensing_method,specifics):
    method = LDAMP_wrapper(sensing_method, specifics)
    return method

def SetNetworkParams(new_height_img, new_width_img,new_channel_img, new_filter_height,new_filter_width,\
                     new_num_filters,new_n_DnCNN_layers,new_n_DAMP_layers, new_sampling_rate,\
                     new_BATCH_SIZE,new_sigma_w,new_n,new_m,new_training, iscomplex,use_adaptive_weights=False):
    global height_img, width_img, channel_img, filter_height, filter_width, num_filters, n_DnCNN_layers, n_DAMP_layers,\
        sampling_rate, BATCH_SIZE, sigma_w, n, m, n_fp, m_fp, is_complex, training, adaptive_weights
    height_img = new_height_img
    width_img = new_width_img
    channel_img = new_channel_img
    filter_height = new_filter_height
    filter_width = new_filter_width
    num_filters = new_num_filters
    n_DnCNN_layers = new_n_DnCNN_layers
    n_DAMP_layers = new_n_DAMP_layers
    sampling_rate = new_sampling_rate
    BATCH_SIZE = new_BATCH_SIZE
    sigma_w = new_sigma_w
    n = new_n
    m = new_m
    n_fp = np.float32(n)
    m_fp = np.float32(m)
    is_complex=iscomplex#Just the default
    adaptive_weights=use_adaptive_weights
    training=new_training

def ListNetworkParameters():
    print ('height_img = ', height_img)
    print ('width_img = ', width_img)
    print ('channel_img = ', channel_img)
    print ('filter_height = ', filter_height)
    print ('filter_width = ', filter_width)
    print ('num_filters = ', num_filters)
    print ('n_DnCNN_layers = ', n_DnCNN_layers)
    print ('n_DAMP_layers = ', n_DAMP_layers)
    print ('sampling_rate = ', sampling_rate)
    print ('BATCH_SIZE = ', BATCH_SIZE)
    print ('sigma_w = ', sigma_w)
    print ('n = ', n)
    print ('m = ', m)
    print('is_complex = ', is_complex)

## Count the total number of learnable parameters
def CountParameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value #TODO # originaly dim.value instead of dim, migration to tensorflow 2.X
        total_parameters += variable_parameters
    print('Total number of parameters: ')
    print(total_parameters)

class LDAMP_wrapper():
    def __init__(self, sensing_method, specifics):
        #Global variables
        self.specifics = specifics
        self.height_img = specifics['height_img']
        self.width_img = specifics['width_img']
        self.channel_img = specifics['channel_img']
        self.filter_height = specifics['filter_height']
        self.filter_width = specifics['filter_width']
        self.num_filters = specifics['num_filters']
        self.n_DnCNN_layers = specifics['n_DnCNN_layers']
        self.sampling_rate = specifics['sampling_rate']
        self.BATCH_SIZE = specifics['BATCH_SIZE']
        self.sigma_w = specifics['sigma_w']
        self.n = specifics['n']
        self.m = specifics['m']
        self.n_fp = np.float32(self.n)
        self.m_fp = np.float32(self.m)

        self.is_complex = False
        if(not(specifics['DenoiserbyDenoiser'])
           and (sensing_method == 'complex-gaussian' or sensing_method == 'coded-diffraction')):
            self.is_complex = True

        self.adaptive_weights = False
        self.training = True

        # used as local variables
        self.start_layer = specifics['start_layer']
        self.max_n_DAMP_layers = specifics['max_n_DAMP_layers']
        self.init_mu = specifics['init_mu']
        self.init_sigma = specifics['init_sigma']
        self.tie_weights = specifics['tie_weights']
        self.LayerbyLayer = specifics['LayerbyLayer']
        self.DenoiserbyDenoiser = specifics['DenoiserbyDenoiser']
        self.sigma_w_min = specifics['sigma_w_min']
        self.sigma_w_max = specifics['sigma_w_max']
        self.alg = specifics['alg']
        self.loss_func = specifics['loss_func']
        self.measurement_mode = specifics['mode']
        self.n_Train_Images = specifics['n_Train_Images']
        self.n_Val_Images = specifics['n_Val_Images']
        self.learning_rates = specifics['learning_rates']
        self.ResumeTraining = specifics['ResumeTraining']
        self.InitWeightsMethod = specifics['InitWeightsMethod']
        self.EPOCHS = specifics['EPOCHS']
        self.max_Epoch_Fails = specifics['max_Epoch_Fails']
        self.validation_patch = specifics['validation_patch']
        self.training_patch = specifics['training_patch']

    def initialize(self,dataset,sensing, stage):
        # do the preparation for the running.
        self.dset = dataset
        self.stage = stage

    def run(self):
        start_layer = self.start_layer
        max_n_DAMP_layers = self.max_n_DAMP_layers
        init_mu = self.init_mu
        init_sigma = self.init_sigma
        tie_weights =  self.tie_weights
        alg = self.alg
        LayerbyLayer = self.LayerbyLayer
        loss_func = self.loss_func
        measurement_mode = self.measurement_mode
        n_Train_Images = self.n_Train_Images
        n_Val_Images = self.n_Val_Images
        learning_rates = self.learning_rates
        ResumeTraining = self.ResumeTraining
        InitWeightsMethod = self.InitWeightsMethod
        EPOCHS = self.EPOCHS
        max_Epoch_Fails = self.max_Epoch_Fails
        stage = self.stage
        validation_patch = self.validation_patch
        training_patch = self.training_patch
        sigma_w_min = self.sigma_w_min
        sigma_w_max = self.sigma_w_max

        train_start_time = time.time()
        print('Denoiser by Denoiser: ', self.DenoiserbyDenoiser)
        print('sudo_rgb: ', self.specifics['sudo_rgb'])
        if(self.DenoiserbyDenoiser):
            if(stage == "training"):
                if loss_func == 'SURE':
                    useSURE = True
                else:
                    useSURE = False

                ## Problem Parameters
                sigma_w_min = sigma_w_min / 255.  # Noise std
                sigma_w_max = sigma_w_max / 255.  # Noise std
                n = self.channel_img * self.height_img * self.width_img

                # Parameters to to initalize weights. Won't be used if old weights are loaded
                init_mu = 0
                init_sigma = 0.1
                ## Clear all the old variables, tensors, etc.
                tf.reset_default_graph()
                SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                 new_sampling_rate=None,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=None, new_n=self.n, new_m=None,
                                 new_training=True, iscomplex=self.is_complex)
                sensing_methods.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                                 new_channel_img=self.channel_img,
                                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                                 new_num_filters=self.num_filters,
                                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                                 new_sampling_rate=None,
                                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=None, new_n=self.n,
                                                 new_m=None, new_training=True, iscomplex=self.is_complex)
                utils.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                       new_channel_img=self.channel_img,
                                       new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                       new_num_filters=self.num_filters,
                                       new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                       new_sampling_rate=None,
                                       new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=None, new_n=self.n, new_m=None,
                                       new_training=True, iscomplex=self.is_complex)
                ListNetworkParameters()

                # tf Graph input
                training_tf = tf.placeholder(tf.bool, name='training')
                sigma_w_tf = tf.placeholder(tf.float32)
                x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

                ## Construct the measurement model and handles/placeholders
                y_measured = utils.AddNoise(x_true, sigma_w_tf)

                ## Initialize the variable theta which stores the weights and biases
                theta_dncnn = init_vars_DnCNN(init_mu, init_sigma)

                ## Construct the reconstruction model
                # x_hat = LDAMP.DnCNN(y_measured,None,theta_dncnn,training=training_tf)
                [x_hat, div_overN] = DnCNN_wrapper(y_measured, None, theta_dncnn, training=training_tf)

                ## Define loss and optimizer

                nfp = np.float32(height_img * width_img)
                if useSURE:
                    cost = utils.MCSURE_loss(x_hat, div_overN, y_measured, sigma_w_tf)
                else:
                    cost = tf.nn.l2_loss(x_true - x_hat) * 1. / nfp

                CountParameters()

                # TODO major change
                # ## Load and Preprocess Training Data
                train_images, val_images = utils.splitDataset(self.dset, self.specifics)
                len_train = len(train_images)
                len_val = len(val_images)

                x_train = np.transpose(np.reshape(train_images, (-1, channel_img * height_img * width_img)))
                x_val = np.transpose(np.reshape(val_images, (-1, channel_img * height_img * width_img)))

                ## Train the Model
                for learning_rate in learning_rates:
                    optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Train all the variables
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        # Ensures that we execute the update_ops before performing the train_step. Allows us to update averages w/in BN
                        optimizer = optimizer0.minimize(cost)

                    saver_best = tf.train.Saver()  # defaults to saving all variables
                    saver_dict = {}

                    config = tf.ConfigProto(allow_soft_placement=True)

                    #TODO This is used to accommodate our RTX graphics card, don't need it otherwise
                    config.gpu_options.allow_growth = True

                    with tf.Session(config=config) as sess:
                        sess.run(
                            tf.global_variables_initializer())  # Seems to be necessary for the batch normalization layers for some reason.

                        # if FLAGS.debug:
                        #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                        #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

                        start_time = time.time()
                        print("Load Initial Weights ...")
                        if ResumeTraining or learning_rate != learning_rates[0]:
                            ##Load previous values for the weights and BNs
                            saver_initvars_name_chckpt = utils.GenDnCNNFilename(sigma_w_min, sigma_w_max,useSURE=useSURE, specifics=self.specifics) + ".ckpt"
                            for l in range(0, n_DnCNN_layers):
                                saver_dict.update({"l" + str(l) + "/w": theta_dncnn[0][l]})
                            for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                gamma_name = "l" + str(l) + "/BN/gamma:0"
                                beta_name = "l" + str(l) + "/BN/beta:0"
                                var_name = "l" + str(l) + "/BN/moving_variance:0"
                                mean_name = "l" + str(l) + "/BN/moving_mean:0"
                                gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                                saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                                saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                                saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                            saver_initvars = tf.train.Saver(saver_dict)
                            saver_initvars.restore(sess, saver_initvars_name_chckpt)
                            # saver_initvars = tf.train.Saver()
                            # saver_initvars.restore(sess, saver_initvars_name_chckpt)
                        else:
                            pass
                        time_taken = time.time() - start_time

                        print("Training ...")
                        print()
                        save_name = utils.GenDnCNNFilename(sigma_w_min, sigma_w_max, useSURE=useSURE, specifics=self.specifics)
                        save_name_chckpt = save_name + ".ckpt"
                        val_values = []
                        print("Initial Weights Validation Value:")
                        rand_inds = np.random.choice(len_val, n_Val_Images, replace=False)
                        start_time = time.time()
                        for offset in range(0, n_Val_Images - BATCH_SIZE + 1,
                                            BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                            end = offset + BATCH_SIZE

                            batch_x_val = x_val[:, rand_inds[offset:end]]
                            sigma_w_thisBatch = sigma_w_min + np.random.rand() * (sigma_w_max - sigma_w_min)

                            # Run optimization.
                            loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, sigma_w_tf: sigma_w_thisBatch,
                                                                 training_tf: False})
                            val_values.append(loss_val)
                        time_taken = time.time() - start_time
                        print(np.mean(val_values))
                        best_val_error = np.mean(val_values)
                        best_sess = sess
                        print("********************")
                        save_path = saver_best.save(best_sess, save_name_chckpt)
                        print("Initial session model saved in file: %s" % save_path)
                        failed_epochs = 0
                        for i in range(EPOCHS):
                            if failed_epochs >= max_Epoch_Fails:
                                break
                            train_values = []
                            print("This Training iteration ...")
                            rand_inds = np.random.choice(len_train, n_Train_Images, replace=False)
                            start_time = time.time()
                            for offset in range(0, n_Train_Images - BATCH_SIZE + 1,
                                                BATCH_SIZE):  # Subtract batch size-1 to avoid errors when len(train_images) is not a multiple of the batch size
                                end = offset + BATCH_SIZE

                                batch_x_train = x_train[:, rand_inds[offset:end]]
                                sigma_w_thisBatch = sigma_w_min + np.random.rand() * (sigma_w_max - sigma_w_min)

                                # Run optimization.
                                _, loss_val = sess.run([optimizer, cost],
                                                       feed_dict={x_true: batch_x_train, sigma_w_tf: sigma_w_thisBatch,
                                                                  training_tf: True})  # Feed dict names should match with the placeholders
                                train_values.append(loss_val)
                            time_taken = time.time() - start_time
                            print(np.mean(train_values))
                            val_values = []
                            print("EPOCH ", i + 1, " Validation Value:")
                            rand_inds = np.random.choice(len_val, n_Val_Images, replace=False)
                            start_time = time.time()
                            for offset in range(0, n_Val_Images - BATCH_SIZE + 1,
                                                BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                                end = offset + BATCH_SIZE

                                batch_x_val = x_val[:, rand_inds[offset:end]]
                                sigma_w_thisBatch = sigma_w_min + np.random.rand() * (sigma_w_max - sigma_w_min)

                                # Run optimization.
                                loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, sigma_w_tf: sigma_w_thisBatch,
                                                                     training_tf: False})
                                val_values.append(loss_val)
                            time_taken = time.time() - start_time
                            print(np.mean(val_values))
                            if (np.mean(val_values) < best_val_error):
                                failed_epochs = 0
                                best_val_error = np.mean(val_values)
                                best_sess = sess
                                print("********************")
                                save_path = saver_best.save(best_sess, save_name_chckpt)
                                print("Best session model saved in file: %s" % save_path)
                            else:
                                failed_epochs = failed_epochs + 1
                            print("********************")

                total_train_time = time.time() - train_start_time
                save_name_time = save_name + "_time.txt"
                # f = open(save_name, 'wb') #TODO convert to python3.7?
                # f.write("Total Training Time =" + str(total_train_time))
                # f.close()
            elif (stage == "testing"):
                train_start_time = time.time()

                ## Clear all the old variables, tensors, etc.
                tf.reset_default_graph()

                SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                 new_sampling_rate=None,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=None,
                                 new_training=False, iscomplex=self.is_complex)
                sensing_methods.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                 new_sampling_rate=None,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=None,
                                 new_training=False, iscomplex=self.is_complex)
                utils.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=None,
                                 new_sampling_rate=None,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=None,
                                 new_training=False, iscomplex=self.is_complex)
                n = self.n
                useSURE = False
                ListNetworkParameters()

                # tf Graph input
                x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

                ## Construct the measurement model and handles/placeholders
                y_measured = utils.AddNoise(x_true, sigma_w)

                ## Initialize the variable theta which stores the weights and biases
                theta_dncnn = init_vars_DnCNN(init_mu, init_sigma)

                ## Construct the reconstruction model
                x_hat = DnCNN(y_measured, None, theta_dncnn, training=False)

                CountParameters()

                # TODO major change
                ## Load and Preprocess Test Data
                test_images = utils.generate_testset(channel_img, width_img, height_img, self.specifics)
                # test_images = test_images[:, 0, :, :]
                assert (len(test_images) >= BATCH_SIZE), "Requested too much Test data"


                x_test = np.transpose(
                    np.reshape(test_images[0:BATCH_SIZE], (BATCH_SIZE, height_img * width_img * channel_img)))
                with tf.Session() as sess:
                    y_test = sess.run(y_measured, feed_dict={x_true: x_test})

                ## Train the Model
                saver = tf.train.Saver()  # defaults to saving all variables
                saver_dict = {}

                config = tf.ConfigProto(allow_soft_placement=True)

                #TODO This is used to accommodate our RTX graphics card, don't need it otherwise
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    # if 255.*sigma_w<10.:
                    #     sigma_w_min=0.
                    #     sigma_w_max=10.
                    # elif 255.*sigma_w<20.:
                    #     sigma_w_min=10.
                    #     sigma_w_max=20.
                    # elif 255.*sigma_w < 40.:
                    #     sigma_w_min = 20.
                    #     sigma_w_max = 40.
                    # elif 255.*sigma_w < 60.:
                    #     sigma_w_min = 40.
                    #     sigma_w_max = 60.
                    # elif 255.*sigma_w < 80.:
                    #     sigma_w_min = 60.
                    #     sigma_w_max = 80.
                    # elif 255.*sigma_w < 100.:
                    #     sigma_w_min = 80.
                    #     sigma_w_max = 100.
                    # elif 255.*sigma_w < 150.:
                    #     sigma_w_min = 100.
                    #     sigma_w_max = 150.
                    # elif 255.*sigma_w < 300.:
                    #     sigma_w_min = 150.
                    #     sigma_w_max = 300.
                    # else:
                    #     sigma_w_min = 300.
                    #     sigma_w_max = 500.
                    # sigma_w_min = sigma_w * 255.
                    # sigma_w_max = sigma_w * 255.

                    save_name = utils.GenDnCNNFilename(sigma_w_min / 255., sigma_w_max / 255., useSURE=useSURE, specifics=self.specifics)
                    save_name_chckpt = save_name + ".ckpt"
                    saver.restore(sess, save_name_chckpt)

                    print("Reconstructing Signal")
                    start_time = time.time()
                    [reconstructed_test_images] = sess.run([x_hat], feed_dict={y_measured: y_test})
                    time_taken = time.time() - start_time

                    # take first image in batch and display
                    if (self.specifics['sudo_rgb']):
                        fig1 = plt.figure()
                        x_recombined = np.reshape(np.transpose(x_test)[:3], (height_img * width_img * 3))
                        plt.imshow(np.reshape(x_recombined, (height_img, width_img, 3)))
                        plt.show()
                        fig2 = plt.figure()
                        x_recombined = np.reshape(np.transpose(y_test)[:3], (height_img * width_img * 3))
                        plt.imshow(np.reshape(x_recombined, (height_img, width_img, 3)))
                        plt.show()
                        fig3 = plt.figure()
                        x_recombined = np.reshape(np.transpose(reconstructed_test_images)[:3],
                                                  (height_img * width_img * 3))
                        plt.imshow(np.reshape(x_recombined, (height_img, width_img, 3)))
                        plt.show()
                        [_, _, PSNR] = utils.EvalError_np(x_test, reconstructed_test_images)
                        print(" PSNR: ", PSNR)
                        print(" Average: ", np.average(PSNR))
                    elif(channel_img == 1):
                        fig1 = plt.figure()
                        plt.imshow(np.transpose(np.reshape(x_test[:, 0], (height_img, width_img))), interpolation='nearest')
                        plt.show()
                        fig2 = plt.figure()
                        plt.imshow(np.transpose(np.reshape(y_test[:, 0], (height_img, width_img))), interpolation='nearest',
                                   cmap='gray')
                        plt.show()
                        fig3 = plt.figure()
                        plt.imshow(np.transpose(np.reshape(reconstructed_test_images[:, 0], (height_img, width_img))),
                                   interpolation='nearest', cmap='gray')
                        plt.show()
                        [_, _, PSNR] = utils.EvalError_np(x_test[:, 0], reconstructed_test_images[:, 0])
                        print(" PSNR: ", PSNR)
                        print(" Average: ", np.average(PSNR))
                    else:
                        fig1 = plt.figure()
                        plt.imshow(np.reshape(x_test[:, 0], (height_img, width_img, 3)))
                        plt.show()
                        fig2 = plt.figure()
                        plt.imshow(np.reshape(y_test[:, 0], (height_img, width_img, 3)))
                        plt.show()
                        fig3 = plt.figure()
                        plt.imshow(np.reshape(reconstructed_test_images[:, 0], (height_img, width_img, 3)))
                        plt.show()
                        [_, _, PSNR] = utils.EvalError_np(x_test, reconstructed_test_images)
                        print(" PSNR: ", PSNR)
                        print(" Average: ", np.average(PSNR))
            else:
                raise Exception("Unknown stage " + stage)

        else:
            if (stage == 'training'):
                for n_DAMP_layers in range(start_layer, max_n_DAMP_layers + 1, 1):
                    ## Clear all the old variables, tensors, etc.
                    tf.reset_default_graph()

                    SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img, new_channel_img=self.channel_img,
                                           new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                           new_num_filters=self.num_filters,
                                           new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                                           new_sampling_rate=self.sampling_rate, new_BATCH_SIZE=self.BATCH_SIZE,
                                            new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m, new_training=True, iscomplex=self.is_complex)
                    sensing_methods.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img, new_channel_img=self.channel_img,
                                           new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                           new_num_filters=self.num_filters,
                                           new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                                           new_sampling_rate=self.sampling_rate, new_BATCH_SIZE=self.BATCH_SIZE,
                                            new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m, new_training=True, iscomplex=self.is_complex)
                    utils.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img, new_channel_img=self.channel_img,
                                           new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                           new_num_filters=self.num_filters,
                                           new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                                           new_sampling_rate=self.sampling_rate, new_BATCH_SIZE=self.BATCH_SIZE,
                                           new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m, new_training=True, iscomplex=self.is_complex)
                    n = self.n
                    ListNetworkParameters()

                    # tf Graph input
                    training_tf = tf.placeholder(tf.bool, name='training')
                    x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

                    ## Initialize the variable theta which stores the weights and biases
                    if tie_weights == True:
                        n_layers_trained = 1
                    else:
                        n_layers_trained = n_DAMP_layers
                    theta = [None] * n_layers_trained
                    for iter in range(n_layers_trained):
                        with tf.variable_scope("Iter" + str(iter)):
                            theta_thisIter = init_vars_DnCNN(init_mu, init_sigma)
                        theta[iter] = theta_thisIter

                    ## Construct the measurement model and handles/placeholders
                    [A_handle, At_handle, A_val, A_val_tf] = sensing_methods.GenerateMeasurementOperators(measurement_mode)
                    y_measured = utils.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)

                    ## Construct the reconstruction model
                    if alg == 'DAMP':
                        (x_hat, MSE_history, NMSE_history, PSNR_history, r_final, rvar_final, div_overN) = LDAMP(
                            y_measured, A_handle, At_handle, A_val_tf, theta, x_true, tie=tie_weights, training=training_tf,
                            LayerbyLayer=LayerbyLayer)
                    elif alg == 'DIT':
                        (x_hat, MSE_history, NMSE_history, PSNR_history) = LDIT(y_measured, A_handle, At_handle, A_val_tf,
                                                                                      theta, x_true, tie=tie_weights,
                                                                                      training=training_tf,
                                                                                      LayerbyLayer=LayerbyLayer)
                    else:
                        raise ValueError('alg was not a supported option')

                    ## Define loss and determine which variables to train
                    nfp = np.float32(height_img * width_img)
                    if loss_func == 'SURE':
                        assert alg == 'DAMP', "Only LDAMP supports training with SURE"
                        cost = utils.MCSURE_loss(x_hat, div_overN, r_final, tf.sqrt(rvar_final))
                    elif loss_func == 'GSURE':
                        assert alg == 'DAMP', "Only LDAMP currently supports training with GSURE"
                        temp0 = tf.matmul(A_val_tf, A_val_tf, transpose_b=True)
                        temp1 = tf.matrix_inverse(temp0)
                        pinv_A = tf.matmul(A_val_tf, temp1, transpose_a=True)
                        P = tf.matmul(pinv_A, A_val_tf)
                        # Treat LDAMP/LDIT as a function of A^ty to calculate the divergence
                        Aty_tf = At_handle(A_val_tf, y_measured)
                        # Overwrite existing x_hat def
                        (x_hat, _, _, _, _, _, _) = LDAMP_Aty(Aty_tf, A_handle, At_handle, A_val_tf, theta, x_true,
                                                                    tie=tie_weights, training=training_tf,
                                                                    LayerbyLayer=LayerbyLayer)
                        if sigma_w == 0.:  # Not sure if TF is smart enough to avoid computing MCdiv when it doesn't have to
                            MCdiv = 0.
                        else:
                            # Calculate MC divergence of P*LDAMP(Aty)
                            epsilon = tf.maximum(.001 * tf.reduce_max(Aty_tf, axis=0), .00001)
                            eta = tf.random_normal(shape=Aty_tf.get_shape(), dtype=tf.float32)
                            Aty_perturbed_tf = Aty_tf + tf.multiply(eta, epsilon)
                            (x_hat_perturbed, _, _, _, _, _, _) = LDAMP_Aty(Aty_perturbed_tf, A_handle, At_handle,
                                                                                  A_val_tf, theta, x_true,
                                                                                  tie=tie_weights, training=training_tf,
                                                                                  LayerbyLayer=LayerbyLayer)
                            Px_hat_perturbed = tf.matmul(P, x_hat_perturbed)
                            Px_hat = tf.matmul(P, x_hat)
                            eta_dx = tf.multiply(eta, Px_hat_perturbed - Px_hat)
                            mean_eta_dx = tf.reduce_mean(eta_dx, axis=0)
                            MCdiv = tf.divide(mean_eta_dx, epsilon) * n
                        x_ML = tf.matmul(pinv_A, y_measured)
                        cost = utils.MCGSURE_loss(x_hat, x_ML, P, MCdiv, sigma_w)
                        # Note: This cost is missing a ||Px||^2 term and so is expected to go negative
                    else:
                        cost = tf.nn.l2_loss(x_true - x_hat) * 1. / nfp

                    iter = n_DAMP_layers - 1
                    if LayerbyLayer == True:
                        vars_to_train = []  # List of only the variables in the last layer.
                        for l in range(0, n_DnCNN_layers):
                            # vars_to_train.extend([theta[iter][0][l], theta[iter][1][l]])
                            vars_to_train.extend([theta[iter][0][l]])
                        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, beta, and gamma
                            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                            beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                            var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                            mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                            vars_to_train.extend([gamma, beta, moving_variance, moving_mean])
                    else:
                        vars_to_train = tf.trainable_variables()

                    CountParameters()
                    #TODO major change
                    # ## Load and Preprocess Training Data
                    train_images, val_images = utils.splitDataset(self.dset, self.specifics)
                    len_train = len(train_images)
                    len_val = len(val_images)

                    x_train = np.transpose(np.reshape(train_images, (-1, channel_img * height_img * width_img)))
                    x_val = np.transpose(np.reshape(val_images, (-1, channel_img * height_img * width_img)))

                    ## Train the Model
                    for learning_rate in learning_rates:
                        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, var_list=vars_to_train)

                        optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            # Ensures that we execute the update_ops before performing the train_step. Allows us to update averages w/in BN
                            optimizer = optimizer0.minimize(cost, var_list=vars_to_train)

                        saver_best = tf.train.Saver()  # defaults to saving all variables
                        saver_dict = {}

                        config = tf.ConfigProto(allow_soft_placement=True)

                        #TODO This is used to accommodate our RTX graphics card, don't need it otherwise
                        config.gpu_options.allow_growth = True

                        with tf.Session(config=config) as sess:
                            sess.run(
                                tf.global_variables_initializer())  # Seems to be necessary for the batch normalization layers for some reason.

                            # if FLAGS.debug:
                            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                            #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

                            start_time = time.time()
                            print("Load Initial Weights ...")
                            if ResumeTraining or learning_rate != learning_rates[0]:
                                ##Load previous values for the weights
                                saver_initvars_name_chckpt = utils.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,loss_func=loss_func, specifics=self.specifics) + ".ckpt"
                                for iter in range(n_layers_trained):  # Create a dictionary with all the variables except those associated with the optimizer.
                                    for l in range(0, n_DnCNN_layers):
                                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][l]})  # ,
                                        # "Iter" + str(iter) + "/l" + str(l) + "/b": theta[iter][1][l]})
                                    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                                        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                                        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                                        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                                        saver_dict.update(
                                            {"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                                        saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                                    saver_initvars = tf.train.Saver(saver_dict)
                                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                                    print("Loaded wieghts from %s" % saver_initvars_name_chckpt)
                            else:
                                ## Load initial values for the weights.
                                # To do so, one associates each variable with a key (e.g. theta[iter][0][0] with l1/w_DnCNN) and loads the l1/w_DCNN weights that were trained on the denoiser
                                # To confirm weights were actually loaded, run sess.run(theta[0][0][0][0][0])[0][0]) before and after this statement. (Requires running sess.run(tf.global_variables_initializer()) first
                                if InitWeightsMethod == 'layer_by_layer':
                                    # load the weights from an identical network that was trained layer-by-layer
                                    saver_initvars_name_chckpt = utils.GenLDAMPFilename(alg, tie_weights, LayerbyLayer=True,loss_func=loss_func, specifics=self.specifics) + ".ckpt"
                                    for iter in range(n_layers_trained):  # Create a dictionary with all the variables except those associated with the optimizer.
                                        for l in range(0, n_DnCNN_layers):
                                            saver_dict.update(
                                                {"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][l]})  # ,
                                            # "Iter" + str(iter) + "/l" + str(l) + "/b": theta[iter][1][l]})
                                        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                                            beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                                            var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                                            mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                                            saver_dict.update(
                                                {"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                                            saver_dict.update(
                                                {"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                                        saver_initvars = tf.train.Saver(saver_dict)
                                        saver_initvars.restore(sess, saver_initvars_name_chckpt)
                                if InitWeightsMethod == 'denoiser':
                                    # load initial weights that were trained on a denoising problem
                                    saver_initvars_name_chckpt = utils.GenDnCNNFilename(300. / 255., 500. / 255., specifics=self.specifics) + ".ckpt"
                                    iter = 0
                                    for l in range(0, n_DnCNN_layers):
                                        saver_dict.update({"l" + str(l) + "/w": theta[iter][0][
                                            l]})  # , "l" + str(l) + "/b": theta[iter][1][l]})
                                    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                                        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                                        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                                        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                        saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                                        saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                                        saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                                        saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                                    saver_initvars = tf.train.Saver(saver_dict)
                                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                                elif InitWeightsMethod == 'smaller_net' and n_DAMP_layers != 1:
                                    # Initialize wieghts using a smaller network's weights
                                    saver_initvars_name_chckpt = utils.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,
                                                                                        n_DAMP_layer_override=n_DAMP_layers - 1,
                                                                                        loss_func=loss_func,
                                                                                        specifics=self.specifics) + ".ckpt"

                                    # Load the first n-1 iterations weights from a previously learned network
                                    for iter in range(n_DAMP_layers - 1):
                                        for l in range(0, n_DnCNN_layers):
                                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][
                                                l]})  # , "Iter"+str(iter)+"/l" + str(l) + "/b": theta[iter][1][l]})
                                        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                            gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                                            beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                                            var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                                            mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                                            saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                                            saver_dict.update(
                                                {"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                                            saver_dict.update(
                                                {"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                                        saver_initvars = tf.train.Saver(saver_dict)
                                        saver_initvars.restore(sess, saver_initvars_name_chckpt)

                                    # Initialize the weights of layer n by using the weights from layer n-1
                                    iter = n_DAMP_layers - 1
                                    saver_dict = {}
                                    for l in range(0, n_DnCNN_layers):
                                        saver_dict.update({"Iter" + str(iter - 1) + "/l" + str(l) + "/w": theta[iter][0][
                                            l]})  # ,"Iter" + str(iter-1) + "/l" + str(l) + "/b": theta[iter][1][l]})
                                    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                                        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                                        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                                        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                                        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                        saver_dict.update({"Iter" + str(iter - 1) + "/l" + str(l) + "/BN/gamma": gamma})
                                        saver_dict.update({"Iter" + str(iter - 1) + "/l" + str(l) + "/BN/beta": beta})
                                        saver_dict.update(
                                            {"Iter" + str(iter - 1) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                                        saver_dict.update(
                                            {"Iter" + str(iter - 1) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
                                    saver_initvars = tf.train.Saver(saver_dict)
                                    saver_initvars.restore(sess, saver_initvars_name_chckpt)
                                else:
                                    # use random weights. This will occur for 1 layer networks if set to use smaller_net initialization
                                    pass
                            time_taken = time.time() - start_time

                            print("Training ...")
                            print()
                            save_name = utils.GenLDAMPFilename(alg, tie_weights, LayerbyLayer, loss_func=loss_func, specifics=self.specifics)
                            save_name_chckpt = save_name + ".ckpt"
                            val_values = []
                            print("Initial Weights Validation Value:")
                            rand_inds = np.random.choice(len_val, n_Val_Images, replace=False)
                            start_time = time.time()
                            for offset in range(0, n_Val_Images - BATCH_SIZE + 1,
                                                BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                                end = offset + BATCH_SIZE

                                # Generate a new measurement matrix
                                A_val = sensing_methods.GenerateMeasurementMatrix(measurement_mode)
                                batch_x_val = x_val[:, rand_inds[offset:end]]

                                # Run optimization. This will both generate compressive measurements and then recontruct from them.
                                loss_val = sess.run(cost,
                                                    feed_dict={x_true: batch_x_val, A_val_tf: A_val, training_tf: False})
                                val_values.append(loss_val)
                            time_taken = time.time() - start_time
                            print(np.mean(val_values))
                            if not LayerbyLayer:  # For end-to-end training save the initial state so that LDAMP end-to-end doesn't diverge when using a high training rate
                                best_val_error = np.mean(val_values)
                                best_sess = sess
                                print("********************")
                                save_path = saver_best.save(best_sess, save_name_chckpt)
                                print("Initial session model saved in file: %s" % save_path)
                            else:  # For layerbylayer training don't save the initial state. With LDIT the initial validation error was often better than the validation error after training 1 epoch. This caused the network not to update and eventually diverge as it got longer and longer
                                best_val_error = np.inf
                            failed_epochs = 0
                            for i in range(EPOCHS):
                                if failed_epochs >= max_Epoch_Fails:
                                    break
                                train_values = []
                                print("This Training iteration ...")
                                rand_inds = np.random.choice(len_train, n_Train_Images, replace=False)
                                start_time = time.time()
                                for offset in range(0, n_Train_Images - BATCH_SIZE + 1,
                                                    BATCH_SIZE):  # Subtract batch size-1 to avoid errors when len(train_images) is not a multiple of the batch size
                                    end = offset + BATCH_SIZE

                                    # Generate a new measurement matrix
                                    A_val = sensing_methods.GenerateMeasurementMatrix(measurement_mode)
                                    batch_x_train = x_train[:, rand_inds[offset:end]]

                                    # Run optimization. This will both generate compressive measurements and then recontruct from them.
                                    _, loss_val = sess.run([optimizer, cost],
                                                           feed_dict={x_true: batch_x_train, A_val_tf: A_val,
                                                                      training_tf: True})  # Feed dict names should match with the placeholders
                                    train_values.append(loss_val)
                                time_taken = time.time() - start_time
                                print(np.mean(train_values))
                                val_values = []
                                print("EPOCH ", i + 1, " Validation Value:")
                                rand_inds = np.random.choice(len_val, n_Val_Images, replace=False)
                                start_time = time.time()
                                for offset in range(0, n_Val_Images - BATCH_SIZE + 1,
                                                    BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                                    end = offset + BATCH_SIZE

                                    # Generate a new measurement matrix
                                    A_val = sensing_methods.GenerateMeasurementMatrix(measurement_mode)
                                    batch_x_val = x_val[:, rand_inds[offset:end]]

                                    # Run optimization. This will both generate compressive measurements and then recontruct from them.
                                    loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, A_val_tf: A_val,
                                                                         training_tf: False})
                                    val_values.append(loss_val)
                                time_taken = time.time() - start_time
                                print(np.mean(val_values))
                                if (np.mean(val_values) < best_val_error):
                                    failed_epochs = 0
                                    best_val_error = np.mean(val_values)
                                    best_sess = sess
                                    print("********************")
                                    save_path = saver_best.save(best_sess, save_name_chckpt)
                                    print("Best session model saved in file: %s" % save_path)
                                else:
                                    failed_epochs = failed_epochs + 1
                                print("********************")
                                total_train_time = time.time() - train_start_time
                                print("Training time so far: %.2f seconds" % total_train_time)
                save_name_txt = save_name + ".txt"
                # f = open(save_name_txt, 'wb') #TODO convert to python3.7?
                # f.write("Total Training Time =" + str(total_train_time))
                # f.close()
            elif(stage == 'testing'):
                ## Testing/Problem Parameters
                # BATCH_SIZE = 1  # Using a batch size larger than 1 will hurt the denoiser by denoiser trained network because it will use an average noise level, rather than a noise level specific to each image
                # n_Test_Images = 5
                # sampling_rate_test = .2  # The sampling rate used for testing
                # sampling_rate_train = .2  # The sampling rate that was used for training
                # sigma_w = 0.
                # n = channel_img * height_img * width_img
                # m = int(np.round(sampling_rate_test * n))
                # measurement_mode = 'Fast-JL'  # 'coded-diffraction'#'gaussian'#'complex-gaussian'#

                # Parameters to to initalize weights. Won't be used if old weights are loaded
                init_mu = 0
                init_sigma = 0.1

                random.seed(1)

                SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=self.max_n_DAMP_layers,
                                 new_sampling_rate=self.sampling_rate,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m,
                                 new_training=False, use_adaptive_weights=self.DenoiserbyDenoiser, iscomplex=self.is_complex)
                sensing_methods.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=self.max_n_DAMP_layers,
                                 new_sampling_rate=self.sampling_rate,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m,
                                 new_training=False, use_adaptive_weights=self.DenoiserbyDenoiser, iscomplex=self.is_complex)
                utils.SetNetworkParams(new_height_img=self.height_img, new_width_img=self.width_img,
                                 new_channel_img=self.channel_img,
                                 new_filter_height=self.filter_height, new_filter_width=self.filter_width,
                                 new_num_filters=self.num_filters,
                                 new_n_DnCNN_layers=self.n_DnCNN_layers, new_n_DAMP_layers=self.max_n_DAMP_layers,
                                 new_sampling_rate=self.sampling_rate,
                                 new_BATCH_SIZE=self.BATCH_SIZE, new_sigma_w=self.sigma_w, new_n=self.n, new_m=self.m,
                                 new_training=False, use_adaptive_weights=self.DenoiserbyDenoiser, iscomplex=self.is_complex)
                n = self.n
                DenoiserbyDenoiser = self.DenoiserbyDenoiser
                n_DAMP_layers = self.max_n_DAMP_layers
                n_Test_Images = self.specifics['n_Test_Images']
                sampling_rate_train = self.sampling_rate
                TrainLoss = self.loss_func
                ListNetworkParameters()

                # tf Graph input
                x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

                # Create handles for the measurement operator
                [A_handle, At_handle, A_val, A_val_tf] = sensing_methods.GenerateMeasurementOperators(measurement_mode)

                ## Initialize the variable theta which stores the weights and biases
                if tie_weights == True:
                    theta = [None]
                    with tf.variable_scope("Iter" + str(0)):
                        theta_thisIter = init_vars_DnCNN(init_mu, init_sigma)
                    theta[0] = theta_thisIter
                elif DenoiserbyDenoiser:
                    noise_min_stds = [0, 10, 20, 40, 60, 80, 100, 150,
                                      300]  # This is currently hardcoded within LearnedDAMP_functionhelper
                    noise_max_stds = [10, 20, 40, 60, 80, 100, 150, 300,
                                      500]  # This is currently hardcoded within LearnedDAMP_functionhelper
                    theta = [None] * len(noise_min_stds)
                    for noise_level in range(len(noise_min_stds)):
                        with tf.variable_scope("Adaptive_NL" + str(noise_level)):
                            theta[noise_level] = init_vars_DnCNN(init_mu, init_sigma)
                else:
                    n_layers_trained = n_DAMP_layers
                    theta = [None] * n_layers_trained
                    for iter in range(n_layers_trained):
                        with tf.variable_scope("Iter" + str(iter)):
                            theta_thisIter = init_vars_DnCNN(init_mu, init_sigma)
                        theta[iter] = theta_thisIter

                ## Construct model
                y_measured = utils.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)
                if alg == 'DAMP':
                    (x_hat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr) = LDAMP(y_measured, A_handle,
                                                                                                  At_handle, A_val_tf,
                                                                                                  theta, x_true,
                                                                                                  tie=tie_weights)
                elif alg == 'DIT':
                    (x_hat, MSE_history, NMSE_history, PSNR_history) = LDIT(y_measured, A_handle, At_handle,
                                                                                  A_val_tf, theta, x_true,
                                                                                  tie=tie_weights)
                else:
                    raise ValueError('alg was not a supported option')

                ## Load and Preprocess Test Data
                # if height_img > 50:
                #     test_im_name = "./TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
                # else:
                #     test_im_name = "./TrainingData/TestData_patch" + str(height_img) + ".npy"
                # test_im_name = self.specifics['testing_patch']
                # TODO major change
                test_images = utils.generate_testset(channel_img, width_img, height_img, self.specifics)
                # test_images = test_images[:n_Test_Images, 0, :, :]
                assert (len(test_images) >= n_Test_Images), "Requested too much Test data"

                x_test = np.transpose(np.reshape(test_images, (-1, height_img * width_img * channel_img)))

                # with tf.Session() as sess:
                #     y_test=sess.run(y_measured,feed_dict={x_true: x_test, A_val_tf: A_val})#All the batches will use the same measurement matrix

                ## Test the Model
                saver = tf.train.Saver()  # defaults to saving all variables
                saver_dict = {}

                #TODO This is used to accommodate our RTX graphics card, don't need it otherwise
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    if tie_weights == 1:  # Load weights from pretrained denoiser
                        save_name = utils.GenDnCNNFilename(80. / 255., specifics=self.specifics) + ".ckpt"
                        for l in range(0, n_DnCNN_layers):
                            saver_dict.update(
                                {"l" + str(l) + "/w": theta[0][0][l]})  # , "l" + str(l) + "/b": theta[0][1][l]})
                        for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                            gamma_name = "Iter" + str(0) + "/l" + str(l) + "/BN/gamma:0"
                            beta_name = "Iter" + str(0) + "/l" + str(l) + "/BN/beta:0"
                            var_name = "Iter" + str(0) + "/l" + str(l) + "/BN/moving_variance:0"
                            mean_name = "Iter" + str(0) + "/l" + str(l) + "/BN/moving_mean:0"
                            gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                            beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                            moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                            moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                            saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                            saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                            saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                            saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                        saver_initvars = tf.train.Saver(saver_dict)
                        saver_initvars.restore(sess, save_name)
                    elif DenoiserbyDenoiser:
                        for noise_level in range(len(noise_min_stds)):
                            noise_min_std = noise_min_stds[noise_level]
                            noise_max_std = noise_max_stds[noise_level]
                            save_name = utils.GenDnCNNFilename(noise_min_std / 255., noise_max_std / 255., specifics=self.specifics) + ".ckpt"
                            for l in range(0, n_DnCNN_layers):
                                saver_dict.update({"l" + str(l) + "/w": theta[noise_level][0][
                                    l]})  # , "l" + str(l) + "/b": theta[noise_level][1][l]})
                            for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                                gamma_name = "Adaptive_NL" + str(noise_level) + "/l" + str(l) + "/BN/gamma:0"
                                beta_name = "Adaptive_NL" + str(noise_level) + "/l" + str(l) + "/BN/beta:0"
                                var_name = "Adaptive_NL" + str(noise_level) + "/l" + str(l) + "/BN/moving_variance:0"
                                mean_name = "Adaptive_NL" + str(noise_level) + "/l" + str(l) + "/BN/moving_mean:0"
                                gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                                beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                                moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                                moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                                saver_dict.update({"l" + str(l) + "/BN/gamma": gamma})
                                saver_dict.update({"l" + str(l) + "/BN/beta": beta})
                                saver_dict.update({"l" + str(l) + "/BN/moving_variance": moving_variance})
                                saver_dict.update({"l" + str(l) + "/BN/moving_mean": moving_mean})
                            saver_initvars = tf.train.Saver(saver_dict)
                            saver_initvars.restore(sess, save_name)
                    else:
                        # save_name = LDAMP.GenLDAMPFilename(alg, tie_weights, LayerbyLayer) + ".ckpt"
                        save_name = utils.GenLDAMPFilename(alg, tie_weights, LayerbyLayer,
                                                           sampling_rate_override=sampling_rate_train,
                                                           loss_func=TrainLoss,
                                                           specifics=self.specifics) + ".ckpt"
                        saver.restore(sess, save_name)

                    print("Reconstructing Signal")
                    start_time = time.time()

                    Final_PSNRs = []
                    Final_SSIMs = []
                    Final_TIMEs = []
                    Final_RECs = []
                    for offset in range(0, n_Test_Images - BATCH_SIZE + 1,
                                        BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                        end = offset + BATCH_SIZE
                        # batch_y_test = y_test[:, offset:end] #To be used when using precomputed measurements

                        # Generate a new measurement matrix
                        A_val = sensing_methods.GenerateMeasurementMatrix(measurement_mode)

                        batch_x_test = x_test[:, offset:end]
                        start = time.time()
                        # Run optimization. This will both generate compressive measurements and then recontruct from them.
                        batch_x_recon, batch_MSE_hist, batch_NMSE_hist, batch_PSNR_hist = sess.run(
                            [x_hat, MSE_history, NMSE_history, PSNR_history],
                            feed_dict={x_true: batch_x_test, A_val_tf: A_val})
                        end = time.time()
                        Final_PSNRs.append(batch_PSNR_hist[-1][0])
                        # Final_PSNRs.append(utils.psnr(batch_x_recon*255, batch_x_test*255))
                        rec_SSIM = sum([ ssim(np.reshape(batch_x_recon[:,i], (height_img, width_img)), np.reshape(batch_x_test[:,i], (height_img, width_img)), data_range=1.0) for i in range(BATCH_SIZE)])/BATCH_SIZE
                        Final_SSIMs.append(rec_SSIM)
                        Final_TIMEs.append(end - start)
                        for i in range(BATCH_SIZE):
                            Final_RECs.append(batch_x_recon[:,i])

                    for i in range(len(Final_PSNRs)):
                        if(Final_PSNRs[i] >= 48.0):
                            Final_PSNRs[i] = 48.0
                    print("PSNR/SSIM/TIME = %.4f/%.4f/%.4f" % (np.mean(Final_PSNRs), np.mean(Final_SSIMs), np.mean(Final_TIMEs)))

                    # fig1 = plt.figure()
                    # plt.imshow(np.transpose(np.reshape(x_test[:, n_Test_Images - 1], (height_img, width_img))),
                    #            interpolation='nearest', cmap='gray')
                    # plt.show()
                    # fig2 = plt.figure()
                    # plt.imshow(np.transpose(np.reshape(batch_x_recon[:, 0], (height_img, width_img))),
                    #            interpolation='nearest', cmap='gray')
                    # plt.show()
                    # fig3 = plt.figure()
                    # plt.plot(range(n_DAMP_layers + 1), np.mean(batch_PSNR_hist, axis=1))
                    # plt.title("PSNR over " + str(alg) + " layers")
                    # plt.show()
                    if (self.specifics['sudo_rgb']):
                        fig1 = plt.figure()
                        x_recombined1 = np.transpose(np.reshape(x_test[:, n_Test_Images - 1], (height_img, width_img)))
                        x_recombined2 = np.transpose(np.reshape(x_test[:, n_Test_Images - 2], (height_img, width_img)))
                        x_recombined3 = np.transpose(np.reshape(x_test[:, n_Test_Images - 3], (height_img, width_img)))
                        x_recombined = [x_recombined1,x_recombined2,x_recombined3]
                        x_recombined = np.array(np.transpose(x_recombined))
                        plt.imshow(x_recombined)
                        plt.show()

                        fig2 = plt.figure()
                        # x_recombined = np.reshape(np.transpose(batch_x_recon)[:3], (height_img * width_img * 3))
                        x_recombined1 = np.transpose(np.reshape(Final_RECs[n_Test_Images - 1], (height_img, width_img)))
                        x_recombined2 = np.transpose(np.reshape(Final_RECs[n_Test_Images - 2], (height_img, width_img)))
                        x_recombined3 = np.transpose(np.reshape(Final_RECs[n_Test_Images - 3], (height_img, width_img)))
                        x_recombined = [x_recombined1, x_recombined2, x_recombined3]
                        x_recombined = np.array(np.transpose(x_recombined))
                        plt.imshow(x_recombined)
                        plt.show()

                        fig3 = plt.figure()
                        plt.plot(range(n_DAMP_layers + 1), np.mean(batch_PSNR_hist, axis=1))
                        plt.title("PSNR over " + str(alg) + " layers")
                        plt.show()
                    elif(channel_img == 1):
                        fig1 = plt.figure()
                        plt.imshow(np.transpose(np.reshape(x_test[:, n_Test_Images - 1], (height_img, width_img))),
                                   interpolation='nearest')
                        plt.show()
                        fig2 = plt.figure()
                        plt.imshow(np.transpose(np.reshape(batch_x_recon[:, BATCH_SIZE - 1], (height_img, width_img))),
                                   interpolation='nearest',
                                   cmap='gray')
                        plt.show()
                        fig3 = plt.figure()
                        plt.plot(range(n_DAMP_layers + 1), np.mean(batch_PSNR_hist, axis=1))
                        plt.title("PSNR over " + str(alg) + " layers")
                        plt.show()
                    else:
                        fig1 = plt.figure()
                        plt.imshow(x_test[:, n_Test_Images - 1], (height_img, width_img, 3))
                        plt.show()
                        fig2 = plt.figure()
                        plt.imshow(np.reshape(batch_x_recon[:, 0], (height_img, width_img, 3)))
                        plt.show()
                        fig3 = plt.figure()
                        plt.show()
                        fig3 = plt.figure()
                        plt.plot(range(n_DAMP_layers + 1), np.mean(batch_PSNR_hist, axis=1))
                        plt.title("PSNR over " + str(alg) + " layers")
                        plt.show()
            else:
                raise Exception("Unknown stage " + stage)

__author__ = 'cmetzler&alimousavi'




#Learned DAMP
def LDAMP(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    z = y
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]#Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=utils.EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        if is_complex:
            r = tf.complex(xhat,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)) + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            r = xhat + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat,dxdr)=DnCNN_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        if is_complex:
            z = y - A_handle(A_val, xhat) + n_fp / m_fp * tf.complex(dxdr,0.) * z
        else:
            z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = utils.EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr

#Learned DAMP operating on Aty. Used for calculating MCSURE loss
def LDAMP_Aty(Aty,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    Atz=Aty
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]#Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=utils.EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        if is_complex:
            r = tf.complex(xhat, tf.zeros([n, BATCH_SIZE], dtype=tf.float32)) + Atz
            rvar = (1. / n_fp * tf.reduce_sum(tf.square(tf.abs(Atz)), axis=0))
            # rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            r = xhat + Atz
            rvar = (1. / n_fp * tf.reduce_sum(tf.square(tf.abs(Atz)), axis=0))
            # rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat,dxdr)=DnCNN_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        if is_complex:
            # z = y - A_handle(A_val, xhat) + n_fp / m_fp * tf.complex(dxdr,0.) * z
            Atz = Aty - At_handle(A_val, A_handle(A_val, xhat)) + n_fp / m_fp * tf.complex(dxdr,0.) * Atz
        else:
            # z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
            Atz = Aty - At_handle(A_val, A_handle(A_val, xhat)) +  n_fp / m_fp * dxdr * Atz
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = utils.EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr

#Learned DIT
def LDIT(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    z=y
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]#Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=utils.EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        if is_complex:
            r = tf.complex(xhat,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)) + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))#In the latest version of TF, abs can handle complex values
        else:
            r = xhat + At_handle(A_val,z)
            rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat, dxdr) = DnCNN_outer_wrapper(r, 4.*rvar, theta, tie, iter,training=training,LayerbyLayer=LayerbyLayer)
        if is_complex:
            z = y - A_handle(A_val, xhat)
        else:
            z = y - A_handle(A_val, xhat)
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = utils.EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history

#Learned DGAMP
def LDGAMP(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True):
    # GAMP notation is used here
    # LDGAMP does not presently support function handles: It does not work with the latest version of the code.
    wvar = tf.square(sigma_w)#Assume noise level is known. Could be learned
    Beta = 1.0 # For now perform no damping
    xhat = tf.zeros([n,BATCH_SIZE], dtype=tf.float32)
    xbar = xhat
    xvar = tf.ones((1,BATCH_SIZE), dtype=tf.float32)
    s = .00001*tf.ones((m, BATCH_SIZE), dtype=tf.float32)
    svar = .00001 * tf.ones((1, BATCH_SIZE), dtype=tf.float32)
    pvar = .00001 * tf.ones((1, BATCH_SIZE), dtype=tf.float32)
    OneOverM = tf.constant(float(1) / m, dtype=tf.float32)
    A_norm2=tf.reduce_sum(tf.square(A_val))
    OneOverA_norm2 = 1. / A_norm2
    MSE_history=[]#Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history=[]
    PSNR_history=[]
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter)=utils.EvalError(xhat,x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        pvar = Beta * A_norm2 * OneOverM * xvar + (1 - Beta) * pvar
        pvar = tf.maximum(pvar, .00001)
        p = tf.matmul(A_val, xhat) - pvar * s
        g, dg = utils.g_out_gaussian(p, pvar, y, wvar)
        s = Beta * g + (1 - Beta) * s
        svar = -Beta * dg + (1 - Beta) * svar
        svar = tf.maximum(svar, .00001)
        rvar = OneOverA_norm2 * n / svar
        rvar = tf.maximum(rvar, .00001)
        xbar = Beta * xhat + (1 - Beta) * xbar
        r = xbar + rvar * tf.matmul(A_val, s,adjoint_a=True)
        (xhat, dxdr) = DnCNN_outer_wrapper(r, rvar, theta, tie, iter,training=training,LayerbyLayer=LayerbyLayer)
        xvar = dxdr * rvar
        xvar = tf.maximum(xvar, .00001)
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = utils.EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history

def init_vars_DnCNN(init_mu,init_sigma):
    #Does not init BN variables
    weights = [None] * n_DnCNN_layers
    biases = [None] * n_DnCNN_layers
    with tf.variable_scope("l0"):
        # Layer 1: filter_heightxfilter_width conv, channel_img inputs, num_filters outputs
        weights[0] = tf.Variable(
            tf.truncated_normal(shape=(filter_height, filter_width, channel_img, num_filters), mean=init_mu,
                                stddev=init_sigma), dtype=tf.float32, name="w")
        #biases[0] = tf.Variable(tf.zeros(num_filters), dtype=tf.float32, name="b")
    for l in range(1, n_DnCNN_layers - 1):
        with tf.variable_scope("l" + str(l)):
            # Layers 2 to Last-1: filter_heightxfilter_width conv, num_filters inputs, num_filters outputs
            weights[l] = tf.Variable(
                tf.truncated_normal(shape=(filter_height, filter_width, num_filters, num_filters), mean=init_mu,
                                    stddev=init_sigma), dtype=tf.float32, name="w")
            #biases[l] = tf.Variable(tf.zeros(num_filters), dtype=tf.float32, name="b")#Need to initialize this with a nz value
            #tf.layers.batch_normalization(inputs=tf.placeholder(tf.float32,[BATCH_SIZE,height_img,width_img,num_filters],name='IsThisIt'), training=tf.placeholder(tf.bool), name='BN', reuse=False)

    with tf.variable_scope("l" + str(n_DnCNN_layers - 1)):
        # Last Layer: filter_height x filter_width conv, num_filters inputs, 1 outputs
        weights[n_DnCNN_layers - 1] = tf.Variable(
            tf.truncated_normal(shape=(filter_height, filter_width, num_filters, 1), mean=init_mu,
                                stddev=init_sigma), dtype=tf.float32,
            name="w")  # The intermediate convolutional layers act on num_filters_inputs, not just channel_img inputs.
        #biases[n_DnCNN_layers - 1] = tf.Variable(tf.zeros(1), dtype=tf.float32, name="b")
    return weights, biases#, betas, moving_variances, moving_means

## Denoiser wrapper that selects which weights and biases to use
def DnCNN_outer_wrapper(r,rvar,theta,tie,iter,training=False,LayerbyLayer=True):
    if tie:
        with tf.variable_scope("Iter0"):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[0], training=training)
    elif adaptive_weights:
        rstd = 255. * tf.sqrt(tf.reduce_mean(rvar))  # To enable batch processing, I have to treat every image in the batch as if it has the same amount of effective noise
        def x_nl0(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(0)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[0], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 0")
            return (xhat, dxdr)

        def x_nl1(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(1)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[1], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 1")
            return (xhat, dxdr)

        def x_nl2(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(2)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[2], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 2")
            return (xhat, dxdr)

        def x_nl3(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(3)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[3], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 3")
            return (xhat, dxdr)

        def x_nl4(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(4)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[4], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 4")
            return (xhat, dxdr)

        def x_nl5(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(5)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[5], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 5")
            return (xhat, dxdr)

        def x_nl6(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(6)):
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[6], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 6")
            return (xhat, dxdr)

        def x_nl7(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(7)) as scope:
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[7], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 7")
            return (xhat, dxdr)

        def x_nl8(a=rstd,iter=iter,r=r,rvar=rvar,theta=theta):
            with tf.variable_scope("Adaptive_NL" + str(8)) as scope:
                (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[8], training=training)
            xhat = tf.Print(xhat, [iter], "used denoiser 8")
            return (xhat, dxdr)

        rstd=tf.Print(rstd,[rstd],"rstd =")
        NL_0 = tf.less_equal(rstd, 10.)
        NL_1 = tf.logical_and(tf.less(10., rstd), tf.less_equal(rstd, 20.))
        NL_2 = tf.logical_and(tf.less(20., rstd), tf.less_equal(rstd, 40.))
        NL_3 = tf.logical_and(tf.less(40., rstd), tf.less_equal(rstd, 60.))
        NL_4 = tf.logical_and(tf.less(60., rstd), tf.less_equal(rstd, 80.))
        NL_5 = tf.logical_and(tf.less(80., rstd), tf.less_equal(rstd, 100.))
        NL_6 = tf.logical_and(tf.less(100., rstd), tf.less_equal(rstd, 150.))
        NL_7 = tf.logical_and(tf.less(150., rstd), tf.less_equal(rstd, 300.))
        predicates = {NL_0:  x_nl0, NL_1:  x_nl1, NL_2:  x_nl2, NL_3:  x_nl3, NL_4:  x_nl4, NL_5:  x_nl5, NL_6:  x_nl6, NL_7:  x_nl7}
        default =  x_nl8
        (xhat,dxdr) = tf.case(predicates,default,exclusive=True)
        xhat = tf.reshape(xhat, shape=[n, BATCH_SIZE])
        dxdr = tf.reshape(dxdr, shape=[1, BATCH_SIZE])
    else:
        with tf.variable_scope("Iter" + str(iter)):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[iter], training=training,LayerbyLayer=LayerbyLayer)
    return (xhat, dxdr)

## Denoiser Wrapper that computes divergence
def DnCNN_wrapper(r,rvar,theta_thislayer,training=False,LayerbyLayer=True):
    """
    Call a black-box denoiser and compute a Monte Carlo estimate of dx/dr
    """
    xhat=DnCNN(r,rvar,theta_thislayer,training=training)
    r_abs = tf.abs(r, name=None)
    epsilon = tf.maximum(.001 * tf.reduce_max(r_abs, axis=0),.00001)
    eta=tf.random_normal(shape=r.get_shape(),dtype=tf.float32)
    if is_complex:
        r_perturbed = r + tf.complex(tf.multiply(eta, epsilon),tf.zeros([n,BATCH_SIZE],dtype=tf.float32))
    else:
        r_perturbed = r + tf.multiply(eta, epsilon)
    xhat_perturbed=DnCNN(r_perturbed,rvar,theta_thislayer,training=training)#Avoid computing gradients wrt this use of theta_thislayer
    eta_dx=tf.multiply(eta,xhat_perturbed-xhat)#Want element-wise multiplication
    mean_eta_dx=tf.reduce_mean(eta_dx,axis=0)
    dxdrMC=tf.divide(mean_eta_dx,epsilon)
    if not LayerbyLayer:
        dxdrMC=tf.stop_gradient(dxdrMC)#When training long networks end-to-end propagating wrt the MC estimates caused divergence
    return(xhat,dxdrMC)

## Create Denoiser Model
def DnCNN(r,rvar, theta_thislayer,training=False):
    #Reuse will always be true, thus init_vars_DnCNN must be called within the appropriate namescope before DnCNN can be used
    ##r is n x batch_size, where in this case n would be height_img*width_img*channel_img
    #rvar is unused within DnCNN. It may have been used to select which sets of weights and biases to use.
    weights=theta_thislayer[0]
    #biases=theta_thislayer[1]

    if is_complex:
        r=tf.real(r)
    r=tf.transpose(r)
    orig_Shape = tf.shape(r)
    shape4D = [-1, height_img, width_img, channel_img]
    r = tf.reshape(r, shape4D)  # reshaping input
    layers = [None] * n_DnCNN_layers

    #############  First Layer ###############
    # Conv + Relu
    with tf.variable_scope("l0"):
        conv_out = tf.nn.conv2d(r, weights[0], strides=[1, 1, 1, 1], padding='SAME',data_format='NHWC') #NCHW works faster on nvidia hardware, however I only perform this type of conovlution once so performance difference will be negligible
        layers[0] = tf.nn.relu(conv_out)

    #############  2nd to 2nd to Last Layer ###############
    # Conv + BN + Relu
    for i in range(1,n_DnCNN_layers-1):
        with tf.variable_scope("l" + str(i)):
            conv_out  = tf.nn.conv2d(layers[i-1], weights[i], strides=[1, 1, 1, 1], padding='SAME') #+ biases[i]
            batch_out = tf.layers.batch_normalization(inputs=conv_out, training=training, name='BN', reuse=tf.AUTO_REUSE)
            layers[i] = tf.nn.relu(batch_out)

    #############  Last Layer ###############
    # Conv
    with tf.variable_scope("l" + str(n_DnCNN_layers - 1)):
        layers[n_DnCNN_layers-1]  = tf.nn.conv2d(layers[n_DnCNN_layers-2], weights[n_DnCNN_layers-1], strides=[1, 1, 1, 1], padding='SAME')

    x_hat = r-layers[n_DnCNN_layers-1]
    x_hat = tf.transpose(tf.reshape(x_hat,orig_Shape))
    return x_hat

