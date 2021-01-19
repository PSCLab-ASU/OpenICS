#TODO tensorflow version 2.X migration code changed the import tensorflow as tf line to two lines as seen below
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf

import numpy as np
import math
import os
import glob
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def SetNetworkParams(new_height_img, new_width_img,new_channel_img, new_filter_height,new_filter_width,\
                     new_num_filters,new_n_DnCNN_layers,new_n_DAMP_layers, new_sampling_rate,\
                     new_BATCH_SIZE,new_sigma_w,new_n,new_m,new_training, iscomplex, use_adaptive_weights=False):
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

def generate_dataset(dataset,input_channel,input_width,input_height,stage, specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(specifics['create_new_dataset'] == False):
        dataset_location = specifics['training_patch']
        data = np.load(dataset_location)
    else:
        if os.path.exists('./Data/' + specifics['dataset_custom_name'] + '.npy'):
            data = np.load('./Data/' + specifics['dataset_custom_name'] + '.npy')
        else:
            data = []
            if(specifics['custom_type_of_image'] == "bmp"):
                images = glob.glob(specifics['new_data'] + "/*.bmp")
            if (specifics['custom_type_of_image'] == "tif"):
                images = glob.glob(specifics['new_data'] + "/*.tif")
            if (specifics['custom_type_of_image'] == "jpg"):
                images = glob.glob(specifics['new_data'] + "/*.jpg")
            if (specifics['custom_type_of_image'] == "png"):
                images = glob.glob(specifics['new_data'] + "/*.png")
            for i, image in enumerate(images):
                with open(image, 'rb') as file:
                    img = Image.open(file)
                    img = np.array(img)
                    if(specifics['sudo_rgb']):
                        if (i == ((specifics['n_Val_Images']+specifics['n_Train_Images']) / 3)):
                            break
                        # scale to between 0 and 1
                        data.append(img[:, :, 0].reshape((1, input_width, input_height)) / 255)
                        data.append(img[:, :, 1].reshape((1, input_width, input_height)) / 255)
                        data.append(img[:, :, 2].reshape((1, input_width, input_height)) / 255)
                    else:
                        if (i == (specifics['n_Val_Images']+specifics['n_Train_Images'])):
                            break
                        # scale to between 0 and 1
                        img = img.reshape((input_channel, input_width, input_height)) / 255
                        data.append(img)
            data = np.array(data)

            if (not (os.path.exists("./Data"))):
                os.mkdir("./Data")
            np.save(os.path.join('./Data', specifics['dataset_custom_name']), data)
            print("################################################################"
                  + "\nCreated new file: ./Data/" + specifics['dataset_custom_name'] + ".npy"
                  + "\n################################################################\n")
    data = data[:specifics['n_Train_Images'] + specifics['n_Val_Images']]
    return data

def generate_testset(input_channel,input_width,input_height,specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(specifics['create_new_testpatch'] == False):
        dataset_location = specifics['testing_patch']
        data = np.load(dataset_location)
    else:
        if os.path.exists('./Data/' + specifics['testset_custom_name'] + '.npy'):
            data = np.load('./Data/' + specifics['testset_custom_name'] + '.npy')
        else:
            data = []
            if (specifics['testing_data_type'] == "bmp"):
                images = glob.glob(specifics['new_test_data'] + "/*.bmp")
            if (specifics['testing_data_type'] == "tif"):
                images = glob.glob(specifics['new_test_data'] + "/*.tif")
            if (specifics['testing_data_type'] == "jpg"):
                images = glob.glob(specifics['new_test_data'] + "/*.jpg")
            if (specifics['testing_data_type'] == "png"):
                images = glob.glob(specifics['new_test_data'] + "/*.png")
            for image in images:
                with open(image, 'rb') as file:
                    img = Image.open(file)
                    img = np.array(img)
                    if (specifics['sudo_rgb']):
                        # scale to between 0 and 1
                        if (img.ndim == 3):
                            data.append(img[:, :, 0].reshape((1, input_width, input_height)) / 255)
                            data.append(img[:, :, 1].reshape((1, input_width, input_height)) / 255)
                            data.append(img[:, :, 2].reshape((1, input_width, input_height)) / 255)
                    else:
                        # scale to between 0 and 1
                        img = img.reshape((input_channel, input_width, input_height)) / 255
                        data.append(img)
            data = np.array(data)

            if (not (os.path.exists("./Data"))):
                os.mkdir("./Data")
            np.save(os.path.join('./Data', specifics['testset_custom_name']), data)
            print("################################################################"
                  + "\nCreated new file: ./Data/" + specifics['testset_custom_name'] + ".npy"
                  + "\n################################################################\n")
    return data

def splitDataset(dset, specifics):
    # load independently
    if (specifics['use_separate_val_patch'] == True):
        train_images = dset[range(specifics['n_Train_Images']), 0, :, :]
        assert (len(train_images) >= specifics['n_Train_Images']), "Requested too much training data"

        val_images = np.load(specifics['validation_patch'])
        val_images = val_images[:, 0, :, :]
        assert (len(val_images) >= specifics['n_Val_Images']), "Requested too much validation data"

    # else randomly shuffle, then split
    else:
        assert (len(dset) >= (specifics['n_Train_Images'] + specifics['n_Val_Images'])), "Not enough data to split into val and train"
        np.random.shuffle(dset)
        train_images = dset
        train_images = train_images[range(specifics['n_Train_Images']), 0, :, :]

        val_images = dset
        val_images = val_images[-1 * specifics['n_Val_Images']:, 0, :, :]


    return train_images, val_images


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1:
        return 48
    PIXEL_MAX = 255.0

    # 20 * log(MaxPixel) - 10 * log(MSE)

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

## Evaluate Intermediate Error
def EvalError(x_hat,x_true):
    x_hat = tf.clip_by_value(x_hat, clip_value_min=0, clip_value_max=1)
    mse=tf.reduce_mean(tf.square((x_hat-x_true)),axis=0)
    # mse=tf.reduce_mean(tf.square((x_hat-x_true) * 255.),axis=0)
    xnorm2=tf.reduce_mean(tf.square( x_true),axis=0)
    mse_thisiter=mse
    nmse_thisiter=mse/xnorm2
    psnr_thisiter = 10. * tf.log(1. / mse) / tf.log(10.)
    # psnr_thisiter = 20. * tf.log(255. / mse) / tf.log(10.)
    return mse_thisiter, nmse_thisiter, psnr_thisiter

## Evaluate Intermediate Error
def EvalError_np(x_hat,x_true):
    mse=np.mean(np.square(x_hat-x_true),axis=0)
    xnorm2=np.mean(np.square( x_true),axis=0)
    mse_thisiter=mse
    nmse_thisiter=mse/xnorm2
    psnr_thisiter=10.*np.log(1./mse)/np.log(10.)
    return mse_thisiter, nmse_thisiter, psnr_thisiter

## Output Denoiser Gaussian
def g_out_gaussian(phat,pvar,y,wvar):
    g=(y-phat)*1/(pvar+wvar)
    dg=-1/(pvar+wvar)
    return g, dg

## Output Denoiser Rician
def g_out_phaseless(phat,pvar,y,wvar):
    #Untested. To be used with D-prGAMP

    y_abs = y
    phat_abs = tf.abs(phat)
    B = 2.* tf.div(tf.multiply(y_abs,phat_abs),wvar+pvar)
    I1overI0 = tf.minimum(tf.div(B,tf.sqrt(tf.square(B)+4)),tf.div(B,0.5+tf.sqrt(tf.square(B)+0.25)))
    y_sca = tf.div(y_abs,1.+tf.div(wvar,pvar))
    phat_sca = tf.div(phat_abs,1.+tf.div(pvar,wvar))
    zhat = tf.multiply(tf.add(phat_sca,tf.multiply(y_sca,I1overI0)),tf.sign(phat))

    sigma2_z = tf.add(tf.add(tf.square(y_sca),tf.square(phat_sca)),tf.subtract(tf.div(1.+tf.multiply(B,I1overI0),tf.add(tf.div(1.,wvar),tf.div(1.,pvar))),tf.square(tf.abs(zhat))))

    g = tf.multiply(tf.div(1.,pvar),tf.subtract(zhat,phat))
    dg = tf.multiply(tf.div(1.,pvar),tf.subtract(tf.div(tf.reduce_mean(sigma2_z,axis=(1,2,3)),pvar),1))

    return g,dg

## Create training data from images, with tf and function handles
def GenerateNoisyCSData_handles(x,A_handle,sigma_w,A_params):
    y = A_handle(A_params,x)
    y = AddNoise(y,sigma_w)
    return y

## Create training data from images, with tf
def GenerateNoisyCSData(x,A,sigma_w):
    y = tf.matmul(A,x)
    y = AddNoise(y,sigma_w)
    return y

## Create training data from images, with tf
def AddNoise(clean,sigma):
    if is_complex:
        noise_vec = sigma/np.sqrt(2) *( tf.complex(tf.random_normal(shape=clean.shape, dtype=tf.float32),tf.random_normal(shape=clean.shape, dtype=tf.float32)))
    else:
        noise_vec=sigma*tf.random_normal(shape=clean.shape,dtype=tf.float32)
    noisy=clean+noise_vec
    noisy=tf.reshape(noisy,clean.shape)
    return noisy

## Create training data from images, with numpy
def GenerateNoisyCSData_np(x,A,sigma_w):
    y = np.matmul(A,x)
    y = AddNoise_np(y,sigma_w)
    return y

## Create training data from images, with numpy
def AddNoise_np(clean,sigma):
    noise_vec=np.random.randn(clean.size)
    noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
    noisy=clean+noise_vec
    return noisy

##Create a string that generates filenames. Ensures consitency between functions
def GenLDAMPFilename(alg,tie_weights,LayerbyLayer,n_DAMP_layer_override=None,sampling_rate_override=None,loss_func='MSE', specifics=None):
    if n_DAMP_layer_override:
        n_DAMP_layers_save=n_DAMP_layer_override
    else:
        n_DAMP_layers_save=n_DAMP_layers
    if sampling_rate_override:
        sampling_rate_save=sampling_rate_override
    else:
        sampling_rate_save=sampling_rate
    filename = specifics['save_folder_name']
    if not os.path.exists(filename):
        os.makedirs(filename)

    if loss_func=='SURE':
        filename = filename + "/SURE_"+alg+"_" + str(n_DnCNN_layers) + "DnCNNL_" + str(int(n_DAMP_layers_save)) + "DAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    elif loss_func=='GSURE':
        filename = filename + "/GSURE_"+alg+"_" + str(n_DnCNN_layers) + "DnCNNL_" + str(int(n_DAMP_layers_save)) + "DAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    else:
        filename = filename + "/"+alg+"_" + str(n_DnCNN_layers) + "DnCNNL_" + str(int(n_DAMP_layers_save)) + "DAMPL_Tie"+str(tie_weights)+"_LbyL"+str(LayerbyLayer)+"_SR" +str(int(sampling_rate_save*100))
    return filename

##Create a string that generates filenames. Ensures consitency between functions
def GenDnCNNFilename(sigma_w_min,sigma_w_max,useSURE=False, specifics=None):
    filename = specifics['save_folder_name']
    if not os.path.exists(filename):
        os.makedirs(filename)

    if useSURE:
        filename = filename + "/SURE_DnCNN_" + str(n_DnCNN_layers) + "L_sigmaMin" + str(
            int(255. * sigma_w_min)) + "_sigmaMax" + str(int(255. * sigma_w_max))
    else:
        filename = filename + "/DnCNN_" + str(n_DnCNN_layers) + "L_sigmaMin" + str(int(255.*sigma_w_min))+"_sigmaMax" + str(int(255.*sigma_w_max))
    return filename

## Calculate Monte Carlo SURE Loss
def MCSURE_loss(x_hat,div_overN,y,sigma_w):
    return tf.reduce_sum(tf.reduce_sum((y - x_hat) ** 2, axis=0) / n_fp -  sigma_w ** 2 + 2. * sigma_w ** 2 * div_overN)

## Calculate Monte Carlo Generalized SURE Loss (||Px||^2 term ignored below)
def MCGSURE_loss(x_hat,x_ML,P,MCdiv,sigma_w):
    Pxhatnorm2 = tf.reduce_sum(tf.abs(tf.matmul(P, x_hat)) ** 2, axis=0)
    temp0 = tf.multiply(x_hat, x_ML)
    x_hatt_xML = tf.reduce_sum(temp0, axis=0)  # x_hat^t*(A^\dagger y)
    return tf.reduce_sum(Pxhatnorm2+2.*sigma_w**2*MCdiv-2.*x_hatt_xML)

## Calculate Monte Carlo Generalized SURE Loss, ||Px||^2 explicitly added so that estimate tracks MSE
def MCGSURE_loss_oracle(x_hat,x_ML,P,MCdiv,sigma_w,x_true):
    Pxtruenorm2 = tf.reduce_sum(tf.abs(tf.matmul(P, x_true)) ** 2, axis=0)
    return tf.reduce_sum(Pxtruenorm2)+MCGSURE_loss(x_hat,x_ML,P,MCdiv,sigma_w)