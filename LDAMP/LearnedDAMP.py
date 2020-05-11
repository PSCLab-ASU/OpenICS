import numpy as np
import math
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def SetNetworkParams(
    new_height_img,
    new_width_img,
    new_channel_img,
    new_filter_height,
    new_filter_width,
    new_num_filters,
    new_n_DnCNN_layers,
    new_n_DAMP_layers,
    new_sampling_rate,
    new_BATCH_SIZE,
    new_sigma_w,
    new_n,
    new_m,
    new_training,
    use_adaptive_weights=False,
):
    global height_img, width_img, channel_img, filter_height, filter_width, num_filters, n_DnCNN_layers, n_DAMP_layers, sampling_rate, BATCH_SIZE, sigma_w, n, m, n_fp, m_fp, is_complex, training, adaptive_weights
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
    is_complex = False  # Just the default
    adaptive_weights = use_adaptive_weights
    training = new_training


def ListNetworkParameters():
    print("height_img = ", height_img)
    print("width_img = ", width_img)
    print("channel_img = ", channel_img)
    print("filter_height = ", filter_height)
    print("filter_width = ", filter_width)
    print("num_filters = ", num_filters)
    print("n_DnCNN_layers = ", n_DnCNN_layers)
    print("n_DAMP_layers = ", n_DAMP_layers)
    print("sampling_rate = ", sampling_rate)
    print("BATCH_SIZE = ", BATCH_SIZE)
    print("sigma_w = ", sigma_w)
    print("n = ", n)
    print("m = ", m)


def GenerateMeasurementOperators(mode):
    if mode == "gaussian":
        A_val = np.float32(
            1.0 / np.sqrt(m_fp) * np.random.randn(m, n)
        )  # values that parameterize the measurement model.
        # This could be the measurement matrix itself or the random mask with coded diffraction patterns.
        y_measured = torch.tensor([m, None])
        A_val_tf = torch.tensor(
            [m, n]
        )  # A placeholer is used so that the large matrix isn't put into the TF graph (2GB limit)

        def A_handle(A_vals_tf, x):
            return torch.mm(A_vals_tf, x)

        def At_handle(A_vals_tf, z):
            return torch.mm(torch.t(A_vals_tf), z)

    return [A_handle, At_handle, A_val, A_val_tf]


def GenerateMeasurementMatrix(mode):
    if mode == "gaussian":
        A_val = np.float32(
            1.0 / np.sqrt(m_fp) * np.random.randn(m, n)
        )  # values that parameterize the measurement model.
        # This could be the measurement matrix itself or the random mask with coded diffraction patterns.
    return A_val


def LDAMP(
    y, A_handle, At_handle, A_val, theta, x_true, tie, training=False, LayerbyLayer=True
):
    z = y
    xhat = torch.zeros([n, BATCH_SIZE], dtype=torch.float32)
    MSE_history = (
        []
    )  # Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history = []
    PSNR_history = []
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        r = xhat + At_handle(A_val, z)
        rvar = 1.0 / m_fp * torch.sum(z ** 2, dim=0)
        (xhat, dxdr) = DnCNN_outer_wrapper(
            r, rvar, theta, tie, iter, training=training, LayerbyLayer=LayerbyLayer
        )
        z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr


# Learned DAMP operating on Aty. Used for calculating MCSURE loss
def LDAMP_Aty(
    Aty,
    A_handle,
    At_handle,
    A_val,
    theta,
    x_true,
    tie,
    training=False,
    LayerbyLayer=True,
):
    Atz = Aty
    xhat = torch.zeros([n, BATCH_SIZE], dtype=torch.float32)
    MSE_history = (
        []
    )  # Will be a list of n_DAMP_layers+1 lists, each sublist will be of size BATCH_SIZE
    NMSE_history = []
    PSNR_history = []
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    for iter in range(n_DAMP_layers):
        r = xhat + Atz
        rvar = 1.0 / n_fp * torch.sum(Atz ** 2, dim=0)
        # rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat, dxdr) = DnCNN_outer_wrapper(
            r, rvar, theta, tie, iter, training=training, LayerbyLayer=LayerbyLayer
        )
        # z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        Atz = Aty - At_handle(A_val, A_handle(A_val, xhat)) + n_fp / m_fp * dxdr * Atz
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter) = EvalError(xhat, x_true)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr


def init_vars_DnCNN(init_mu, init_sigma):
    # Does not init BN variables
    weights = [None] * n_DnCNN_layers
    biases = [None] * n_DnCNN_layers
    # let's see what happens when we don't scope the variables and then we can replace it
    # Layer 1: filter_heightxfilter_width conv, channel_img inputs, num_filters outputs

    weights[0] = torch.from_numpy(
        truncnorm.rvs(
            init_sigma * -2 + init_mu,
            init_sigma * 2 + init_mu,
            size=[filter_height, filter_width, channel_img, num_filters],
        )
    )
    # biases[0] = tf.Variable(tf.zeros(num_filters), dtype=tf.float32, name="b")
    for l in range(1, n_DnCNN_layers - 1):
        # Layers 2 to Last-1: filter_heightxfilter_width conv, num_filters inputs, num_filters outputs
        weights[l] = torch.from_numpy(
            truncnorm.rvs(
                init_sigma * -2 + init_mu,
                init_sigma * 2 + init_mu,
                size=[filter_height, filter_width, num_filters, num_filters],
            )
        )
        # biases[l] = tf.Variable(tf.zeros(num_filters), dtype=tf.float32, name="b")#Need to initialize this with a nz value
        # tf.layers.batch_normalization(inputs=tf.placeholder(tf.float32,[BATCH_SIZE,height_img,width_img,num_filters],name='IsThisIt'), training=tf.placeholder(tf.bool), name='BN', reuse=False)

    # Last Layer: filter_height x filter_width conv, num_filters inputs, 1 outputs
    weights[n_DnCNN_layers - 1] = torch.from_numpy(
        truncnorm.rvs(
            init_sigma * -2 + init_mu,
            init_sigma * 2 + init_mu,
            size=[filter_height, filter_width, num_filters, 1],
        )
    )  # The intermediate convolutional layers act on num_filters_inputs, not just channel_img inputs.
    # biases[n_DnCNN_layers - 1] = tf.Variable(tf.zeros(1), dtype=tf.float32, name="b")
    return weights, biases  # , betas, moving_variances, moving_means


def EvalError(x_hat, x_true):
    mse = torch.mean((x_hat - x_true) ** 2, dim=0)
    xnorm2 = torch.mean(x_true ** 2, dim=0)
    mse_thisiter = mse
    nmse_thisiter = mse / xnorm2
    psnr_thisiter = 10.0 * math.log(1.0 / mse.item()) / math.log(10.0)
    return mse_thisiter, nmse_thisiter, psnr_thisiter


## Evaluate Intermediate Error
def EvalError_np(x_hat, x_true):
    mse = np.mean(np.square(x_hat - x_true), axis=0)
    xnorm2 = np.mean(np.square(x_true), axis=0)
    mse_thisiter = mse
    nmse_thisiter = mse / xnorm2
    psnr_thisiter = 10.0 * np.log(1.0 / mse) / np.log(10.0)
    return mse_thisiter, nmse_thisiter, psnr_thisiter


## Output Denoiser Gaussian
def g_out_gaussian(phat, pvar, y, wvar):
    g = (y - phat) * 1 / (pvar + wvar)
    dg = -1 / (pvar + wvar)
    return g, dg


## Denoiser wrapper that selects which weights and biases to use
def DnCNN_outer_wrapper(r, rvar, theta, tie, iter, training=False, LayerbyLayer=True):
    if tie:
        (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[0], training=training)
    elif adaptive_weights:
        rstd = 255.0 * torch.sqrt(
            torch.mean(rvar)
        )  # To enable batch processing, I have to treat every image in the
        # batch as if it has the same amount of effective noise

        def x_nl0(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[0], training=training)
            print(xhat, [iter], "used denoiser 0")
            return xhat, dxdr

        def x_nl1(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[1], training=training)
            print(xhat, [iter], "used denoiser 1")
            return xhat, dxdr

        def x_nl2(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[2], training=training)
            print(xhat, [iter], "used denoiser 2")
            return xhat, dxdr

        def x_nl3(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[3], training=training)
            print(xhat, [iter], "used denoiser 3")
            return xhat, dxdr

        def x_nl4(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[4], training=training)
            print(xhat, [iter], "used denoiser 4")
            return xhat, dxdr

        def x_nl5(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[5], training=training)
            print(xhat, [iter], "used denoiser 5")
            return xhat, dxdr

        def x_nl6(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[6], training=training)
            print(xhat, [iter], "used denoiser 6")
            return xhat, dxdr

        def x_nl7(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[7], training=training)
            print(xhat, [iter], "used denoiser 7")
            return xhat, dxdr

        def x_nl8(a=rstd, iter=iter, r=r, rvar=rvar, theta=theta):
            (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[8], training=training)
            print(xhat, [iter], "used denoiser 8")
            return xhat, dxdr

        print(rstd, [rstd], "rstd =")
        NL_0 = rstd <= 10.0
        NL_1 = (10.0 < rstd) & (rstd <= 20.0)
        NL_2 = (20.0 < rstd) & (rstd <= 40.0)
        NL_3 = (40.0 < rstd) & (rstd <= 60.0)
        NL_4 = (60.0 < rstd) & (rstd <= 80.0)
        NL_5 = (80.0 < rstd) & (rstd <= 100.0)
        NL_6 = (100.0 < rstd) & (rstd <= 150.0)
        NL_7 = (150.0 < rstd) & (rstd <= 300.0)
        predicates = {
            NL_0: x_nl0,
            NL_1: x_nl1,
            NL_2: x_nl2,
            NL_3: x_nl3,
            NL_4: x_nl4,
            NL_5: x_nl5,
            NL_6: x_nl6,
            NL_7: x_nl7,
        }
        default = x_nl8
        # TODO
        (xhat, dxdr) = tf.case(predicates, default, exclusive=True)
        xhat = torch.reshape(xhat, [n, BATCH_SIZE])
        dxdr = torch.reshape(dxdr, [1, BATCH_SIZE])
    else:
        (xhat, dxdr) = DnCNN_wrapper(
            r, rvar, theta[iter], training=training, LayerbyLayer=LayerbyLayer
        )
    return (xhat, dxdr)


## Denoiser Wrapper that computes divergence
def DnCNN_wrapper(r, rvar, theta_thislayer, dncnn, dncnnWrapper, training=False, LayerbyLayer=True):
    """
    Call a black-box denoiser and compute a Monte Carlo estimate of dx/dr
    """

    xhat = dncnn.forward(r)
    r_abs = torch.abs(r)

    r_absColumnWise = torch.max(r_abs, dim=0)[0]
    
    epsilon = torch.max(r_absColumnWise*0.001, torch.ones(r_absColumnWise.shape)*0.00001)

    eta = torch.normal(0,1,r.shape)#Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample(r.shape)

    r_perturbed = r + torch.mul(eta, epsilon)

    xhat_perturbed = dncnnWrapper.forward(r_perturbed)

    eta_dx = torch.mul(eta, xhat_perturbed - xhat)  # Want element-wise multiplication
    mean_eta_dx = torch.mean(eta_dx, dim=0)
    dxdrMC = torch.div(mean_eta_dx, epsilon)
    # if not LayerbyLayer:
    #     dxdrMC = tf.stop_gradient(
    #         dxdrMC
    #     )  # When training long networks end-to-end propagating wrt the MC estimates caused divergence
    return (xhat, dxdrMC)


class DnCNN(nn.Module):
        def __init__(self):
            super(DnCNN, self).__init__()
            #self.pad = torch.nn.ZeroPad2d((0,1,0,1))
            self.conv0 = nn.Conv2d(1, 64,kernel_size = 3, stride=1,padding=1)
            self.layers = nn.ModuleList()

            for i in range(1,n_DnCNN_layers-1):              
                #pad = torch.nn.ZeroPad2d((0,1,0,1)) 
                conv = nn.Conv2d(64,64, kernel_size = 3, stride=1,padding=1)
                batchnorm = nn.BatchNorm2d(64) #training
                #self.layers.add_module(pad)
                self.layers.add_module("conv"+str(i), conv)
                self.layers.add_module("batch"+str(i), batchnorm)
            self.convLast = nn.Conv2d(64, 1,kernel_size = 3, stride=1,padding=1)
        def forward(self, r):
            r = torch.t(r)
            orig_Shape = r.shape
            shape4D = [-1, channel_img, width_img, height_img]#shape4D = [-1, height_img, width_img, channel_img]
            r = torch.reshape(r, shape4D)  # reshaping input
            #print("input shape:" + str(r.shape))
            x = F.relu(self.conv0(r))
            #print(x.shape)
            for i, l in enumerate(self.layers): 
                #x = self.pad(x)
                x = l(x)
                #print(x.shape)
            x = F.relu(self.convLast(x))
            x_hat = r-x
            x_hat = torch.t(torch.reshape(x_hat, orig_Shape))
            return x_hat
## Create Denoiser Model

## Create training data from images, with tf and function handles
def GenerateNoisyCSData_handles(x, A_handle, sigma_w, A_params):
    y = A_handle(A_params, x)
    y = AddNoise(y, sigma_w)
    return y


## Create training data from images, with tf
def GenerateNoisyCSData(x, A, sigma_w):
    y = torch.matmul(A, x)
    y = AddNoise(y, sigma_w)
    return y


## Create training data from images, with tf
def AddNoise(clean, sigma):
    noise_vec = sigma * torch.distributions.Normal(0, 1).rsample(
        sample_shape=clean.shape
    )
    noisy = clean + noise_vec
    noisy = torch.reshape(noisy, clean.shape)
    return noisy


## Create training data from images, with numpy
def GenerateNoisyCSData_np(x, A, sigma_w):
    y = np.matmul(A, x)
    y = AddNoise_np(y, sigma_w)
    return y


## Create training data from images, with numpy
def AddNoise_np(clean, sigma):
    noise_vec = np.random.randn(clean.size)
    noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
    noisy = clean + noise_vec
    return noisy


##Create a string that generates filenames. Ensures consitency between functions
def GenLDAMPFilename(
    alg,
    tie_weights,
    LayerbyLayer,
    n_DAMP_layer_override=None,
    sampling_rate_override=None,
    loss_func="MSE",
):
    if n_DAMP_layer_override:
        n_DAMP_layers_save = n_DAMP_layer_override
    else:
        n_DAMP_layers_save = n_DAMP_layers
    if sampling_rate_override:
        sampling_rate_save = sampling_rate_override
    else:
        sampling_rate_save = sampling_rate
    if loss_func == "SURE":
        filename = (
            "./saved_models/LDAMP/SURE_"
            + alg
            + "_"
            + str(n_DnCNN_layers)
            + "DnCNNL_"
            + str(int(n_DAMP_layers_save))
            + "DAMPL_Tie"
            + str(tie_weights)
            + "_LbyL"
            + str(LayerbyLayer)
            + "_SR"
            + str(int(sampling_rate_save * 100))
        )
    elif loss_func == "GSURE":
        filename = (
            "./saved_models/LDAMP/GSURE_"
            + alg
            + "_"
            + str(n_DnCNN_layers)
            + "DnCNNL_"
            + str(int(n_DAMP_layers_save))
            + "DAMPL_Tie"
            + str(tie_weights)
            + "_LbyL"
            + str(LayerbyLayer)
            + "_SR"
            + str(int(sampling_rate_save * 100))
        )
    else:
        filename = (
            "./saved_models/LDAMP/"
            + alg
            + "_"
            + str(n_DnCNN_layers)
            + "DnCNNL_"
            + str(int(n_DAMP_layers_save))
            + "DAMPL_Tie"
            + str(tie_weights)
            + "_LbyL"
            + str(LayerbyLayer)
            + "_SR"
            + str(int(sampling_rate_save * 100))
        )
    return filename


##Create a string that generates filenames. Ensures consitency between functions
def GenDnCNNFilename(sigma_w_min, sigma_w_max, useSURE=False):
    if useSURE:
        filename = (
            "./saved_models/DnCNN/SURE_DnCNN_"
            + str(n_DnCNN_layers)
            + "L_sigmaMin"
            + str(int(255.0 * sigma_w_min))
            + "_sigmaMax"
            + str(int(255.0 * sigma_w_max))
        )
    else:
        filename = (
            "./saved_models/DnCNN/DnCNN_"
            + str(n_DnCNN_layers)
            + "L_sigmaMin"
            + str(int(255.0 * sigma_w_min))
            + "_sigmaMax"
            + str(int(255.0 * sigma_w_max))
        )
    return filename


## Count the total number of learnable parameters
def CountParameters():
    # total_parameters = 0
    # for variable in tf.trainable_variables():
    #     shape = variable.get_shape()
    #     variable_parameters = 1
    #     for dim in shape:
    #         variable_parameters *= dim.value
    #     total_parameters += variable_parameters
    # print("Total number of parameters: ")
    # print(total_parameters)
    # TODO
    print("not implemented")


## Calculate Monte Carlo SURE Loss
def MCSURE_loss(x_hat, div_overN, y, sigma_w):
    return torch.sum(
        torch.sum((y - x_hat) ** 2, dim=0) / n_fp
        - sigma_w ** 2
        + 2.0 * sigma_w ** 2 * div_overN
    )


## Calculate Monte Carlo Generalized SURE Loss (||Px||^2 term ignored below)
def MCGSURE_loss(x_hat, x_ML, P, MCdiv, sigma_w):
    Pxhatnorm2 = torch.sum(torch.abs(torch.matmul(P, x_hat)) ** 2, dim=0)
    temp0 = x_hat * x_ML
    x_hatt_xML = torch.sum(temp0, dim=0)  # x_hat^t*(A^\dagger y)
    return torch.sum(Pxhatnorm2 + 2.0 * sigma_w ** 2 * MCdiv - 2.0 * x_hatt_xML)


## Calculate Monte Carlo Generalized SURE Loss, ||Px||^2 explicitly added so that estimate tracks MSE
def MCGSURE_loss_oracle(x_hat, x_ML, P, MCdiv, sigma_w, x_true):
    Pxtruenorm2 = torch.sum(torch.abs(torch.matmul(P, x_true)) ** 2, dim=0)
    return torch.sum(Pxtruenorm2) + MCGSURE_loss(x_hat, x_ML, P, MCdiv, sigma_w)