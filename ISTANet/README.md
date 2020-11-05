Variables:

sensing: unused
reconstruction: type of reconstruction (IstaNetPlus)
stage: either "training" or "testing"
default: default is training
dataset: name of dataset to train on
input_channel: unused
input_width: unused
input_height: unused
m: unused
n: unused
specifics: (used to store variables conveniently and define new ones)
	stage: taken from original input
	start_epoch: set to non-zero when loading a specific network for training
	end_epoch: total number of epochs used in training
	testing_epoch_num: used when loading a specific network for testing
	learning_rate: learning rate used in training
	layer_num: defines number of BasicBlocks used in the IstaNetPlus
	group_num: used to name the folder containing the network, organizational
	cs_ratio: ratio used in training/testing
	nrtrain: number of images to be trained on, only used for displaying progress
	batch_size: size of batches fed into the network when training/testing (use 64 for training and 1 for testing)
	model_dir: local directory where the model groups will be stored
	log_dir: local directory where the logs will be stored
	data_dir: local directory where the training and testing data will be stored 
	result_dir: local directory where the reconstructed images go
	matrix_dir: local directory where the sampling matrices go
	Training_data_Name: taken from original input, used in utils to retrieve training labels
	test_name: directory where the test images reside


Commands to set up in conda:

conda create -n pytCUDA python=3.7.0
conda install pytorch==1.6.0 torchvision cudatoolkit=10.2 -c pytorch
conda install scipy==1.5.2
conda install opencv==3.4.2
conda install scikit-image==0.17.2


Data:

use Set11 for testing
use Training_Data.mat for training


