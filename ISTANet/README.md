# ISTANETPlus and ISTANET
### Description:
This implementation of ISTANet and ISTANet-Plus is based off of https://github.com/jianzhangcs/ISTA-Net. ISTANet and ISTANet-Plus were merged into the same framework and are both accessible with the provided code. All variables are set through the main.py function.

### Variables:
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
&nbsp;&nbsp;&nbsp;&nbsp;sudo_rgb: instead of setting the channel to 3, set sudo_rgb to True<br />
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
&nbsp;&nbsp;&nbsp;&nbsp;training_data_fileName: the name of the training data file<br />
&nbsp;&nbsp;&nbsp;&nbsp;training_data_type: the type of the training data file<br />
&nbsp;&nbsp;&nbsp;&nbsp;create_custom_dataset: true if you want to create a new dataset<br />
&nbsp;&nbsp;&nbsp;&nbsp;custom_dataset_name: name of the custom dataset to be created<br />
&nbsp;&nbsp;&nbsp;&nbsp;custom_training_data_location: location of the custom dataset<br />
&nbsp;&nbsp;&nbsp;&nbsp;custom_type_of_image: type of image to be parsed in the training data location<br />
&nbsp;&nbsp;&nbsp;&nbsp;Testing_data_location: location of the testing data<br />
&nbsp;&nbsp;&nbsp;&nbsp;testing_data_isFolderImages: true if the location contains raw images<br />
&nbsp;&nbsp;&nbsp;&nbsp;testing_data_type:  type of image to be parsed in the testing data location<br />
