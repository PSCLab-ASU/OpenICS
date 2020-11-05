# Variables:
sensing: unused<br />
reconstruction: type of reconstruction (IstaNetPlus)<br />
stage: either "training" or "testing"<br />
default: default is training<br />
dataset: name of dataset to train on<br />
input_channel: unused<br />
input_width: unused<br />
input_height: unused<br />
m: unused<br />
n: unused<br />
specifics: (used to store variables conveniently and define new ones)<br />
&nbsp;&nbsp;&nbsp;&nbsp;stage: taken from original input<br />
&nbsp;&nbsp;&nbsp;&nbsp;start_epoch: set to non-zero when loading a specific network for training<br />
&nbsp;&nbsp;&nbsp;&nbsp;end_epoch: total number of epochs used in training<br />
&nbsp;&nbsp;&nbsp;&nbsp;testing_epoch_num: used when loading a specific network for testing<br />
&nbsp;&nbsp;&nbsp;&nbsp;learning_rate: learning rate used in training<br />
&nbsp;&nbsp;&nbsp;&nbsp;layer_num: defines number of BasicBlocks used in the IstaNetPlus<br />
&nbsp;&nbsp;&nbsp;&nbsp;group_num: used to name the folder containing the network, organizational<br />
&nbsp;&nbsp;&nbsp;&nbsp;cs_ratio: ratio used in training/testing<br />
&nbsp;&nbsp;&nbsp;&nbsp;nrtrain: number of images to be trained on, only used for displaying progress<br />
&nbsp;&nbsp;&nbsp;&nbsp;batch_size: size of batches fed into the network when training/testing (use 64 for training and 1 for testing)<br />
&nbsp;&nbsp;&nbsp;&nbsp;model_dir: local directory where the model groups will be stored<br />
&nbsp;&nbsp;&nbsp;&nbsp;log_dir: local directory where the logs will be stored<br />
&nbsp;&nbsp;&nbsp;&nbsp;data_dir: local directory where the training and testing data will be stored <br />
&nbsp;&nbsp;&nbsp;&nbsp;result_dir: local directory where the reconstructed images go<br />
&nbsp;&nbsp;&nbsp;&nbsp;matrix_dir: local directory where the sampling matrices go<br />
&nbsp;&nbsp;&nbsp;&nbsp;Training_data_Name: taken from original input, used in utils to retrieve training labels<br />
&nbsp;&nbsp;&nbsp;&nbsp;test_name: directory where the test images reside<br />

# Commands to set up in conda:

conda create -n pytCUDA python=3.7.0<br />
conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch<br />
conda install scipy==1.5.2<br />
conda install opencv==3.4.2<br />
conda install scikit-image==0.17.2<br />

# Data:

use Set11 for testing<br />
use Training_Data.mat for training<br />


