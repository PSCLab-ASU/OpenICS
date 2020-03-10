import numpy as np
import argparse
import torch
import time
import LearnedDAMP as LDAMP
from matplotlib import pyplot as plt
import h5py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## Network Parameters
height_img = 256
width_img = 256
channel_img = 1 # RGB -> 3, Grayscale -> 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=16
useSURE=False#Use the network trained with ground-truth data or with SURE


## Training Parameters
BATCH_SIZE = 1

## Problem Parameters
sigma_w=25./255.#Noise std
n=channel_img*height_img*width_img

# Parameters to to initalize weights. Won't be used if old weights are loaded
init_mu = 0
init_sigma = 0.1

train_start_time=time.time()

## Clear all the old variables, tensors, etc.

LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                       new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                       new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=None,
                       new_sampling_rate=None, \
                       new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=None, new_training=False)
LDAMP.ListNetworkParameters()



LDAMP.CountParameters()

## Load and Preprocess Test Data
if height_img>50:
    test_im_name = "./TrainingData/StandardTestData_" + str(height_img) + "Res.npy"
else:
    test_im_name = "./TrainingData/TestData_patch" + str(height_img) + ".npy"

test_images = np.load(test_im_name)
test_images=test_images[:,0,:,:]
assert (len(test_images)>=BATCH_SIZE), "Requested too much Test data"

with torch.no_grad():
    dncnn = LDAMP.DnCNN(init_mu,init_sigma).to(device)
    dncnn.load_state_dict(torch.load("savedmodels/best.pth"))
    x_test = torch.from_numpy(np.transpose(np.reshape(test_images[0:BATCH_SIZE], (BATCH_SIZE, height_img * width_img * channel_img)))).to(device).float()
    y_test= LDAMP.AddNoise(x_test,sigma_w)
    print('TYPE', x_test.type())
    print("Reconstructing Signal")
    start_time = time.time()
    reconstructed_test_images = dncnn.forward(y_test)
    time_taken=time.time()-start_time
fig1 = plt.figure()
plt.imshow(np.transpose(np.reshape(x_test[:, 0].to("cpu"), (height_img, width_img))), interpolation='nearest', cmap='gray')
plt.show()
fig2 = plt.figure()
plt.imshow(np.transpose(np.reshape(y_test[:, 0].to("cpu"), (height_img, width_img))), interpolation='nearest', cmap='gray')
plt.show()
fig3 = plt.figure()
plt.imshow(np.transpose(np.reshape(reconstructed_test_images[:, 0].to("cpu"), (height_img, width_img))), interpolation='nearest', cmap='gray')
plt.show()
[_,_,PSNR]=LDAMP.EvalError(x_test[:, 0].to("cpu"),reconstructed_test_images[:, 0].to("cpu"))
print(" PSNR: " ,PSNR)
