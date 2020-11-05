# Variables:
sensing: unused<br />
reconstruction: type of reconstruction (LDAMP)<br />
stage: either "training" or "testing"<br />
default: default is training<br />
dataset: name of dataset to train on<br />
input_channel: number of channels the training data has<br />
input_width: width of training data images<br />
input_height: height of training data images<br />
m: manually calculated<br />
n: manually calculated<br />
specifics: (used to store variables conveniently and define new ones)<br />
&nbsp;&nbsp;&nbsp;&nbsp;channel_img: stored from original input <br />
&nbsp;&nbsp;&nbsp;&nbsp;width_img: stored from original input<br />
&nbsp;&nbsp;&nbsp;&nbsp;height_img: stored from original input<br />
&nbsp;&nbsp;&nbsp;&nbsp;n: manually calculated<br />
&nbsp;&nbsp;&nbsp;&nbsp;m: manually calculated<br />
&nbsp;&nbsp;&nbsp;&nbsp;sampling_rate: rate at which the training data is sampled<br />
&nbsp;&nbsp;&nbsp;&nbsp;alg: algorithm used, in this case DAMP<br />
&nbsp;&nbsp;&nbsp;&nbsp;tie_weights: if true, only one layer of LDAMP is trained<br />
&nbsp;&nbsp;&nbsp;&nbsp;filter_height: denoiser filter height<br />
&nbsp;&nbsp;&nbsp;&nbsp;filter_width: denoiser filter width<br />
&nbsp;&nbsp;&nbsp;&nbsp;num_filters: number of filters in each layer of the DnCNN<br />
&nbsp;&nbsp;&nbsp;&nbsp;n_DnCNN_layers: number of denoiser layers<br />
&nbsp;&nbsp;&nbsp;&nbsp;max_n_DAMP_layers: number of LDAMP layers to train to or layer to test at<br />
&nbsp;&nbsp;&nbsp;&nbsp;start_layer: which layer to start on<br />
&nbsp;&nbsp;&nbsp;&nbsp;max_Epoch_Fails: How many training epochs to run without improvement in the validation error<br />
&nbsp;&nbsp;&nbsp;&nbsp;ResumeTraining: option to resume training a network you were previously training<br />
&nbsp;&nbsp;&nbsp;&nbsp;LayerByLayer: true if training layer by layer<br />
&nbsp;&nbsp;&nbsp;&nbsp;DenoiserbyDenoiser: true if training denoiser by denoiser (overrides LayerByLayer)<br />
&nbsp;&nbsp;&nbsp;&nbsp;sigma_w_min: used in denoiesrbydenoiser training, determines noise<br />
&nbsp;&nbsp;&nbsp;&nbsp;sigma_w_max: used in denoiesrbydenoiser training, determines noise<br />
&nbsp;&nbsp;&nbsp;&nbsp;sigma_w: used in both LbL and DbD to determine noise (LbL testing: 0, LbL training: 1./255., DbD test and train: 25./255.)<br />
&nbsp;&nbsp;&nbsp;&nbsp;learning_rates: learning rates used, once one does not yield improvement -> move to next<br />
&nbsp;&nbsp;&nbsp;&nbsp;EPOCHS: number of epochs<br />
&nbsp;&nbsp;&nbsp;&nbsp;n_Train_Images: number of training images used<br />
&nbsp;&nbsp;&nbsp;&nbsp;n_Val_Images: number of validation images used<br />
&nbsp;&nbsp;&nbsp;&nbsp;BATCH_SIZE: batch size used during training and testing (typically 128 for training, 1 for testing)<br />
&nbsp;&nbsp;&nbsp;&nbsp;InitWeightsMethod: way to initialize the weights, (default: smaller_net; options: random, denoiser, smaller net, and layer_by_layer)<br />
&nbsp;&nbsp;&nbsp;&nbsp;loss_func: loss function used, can only use MSE currently<br />
&nbsp;&nbsp;&nbsp;&nbsp;init_mu: mean when initializing denoiser layers<br />
&nbsp;&nbsp;&nbsp;&nbsp;init_sigma: sigma (std deviation) when initializing denoiser layers<br />
&nbsp;&nbsp;&nbsp;&nbsp;mode: sensing mode (default: gaussian)<br />
&nbsp;&nbsp;&nbsp;&nbsp;validation_patch: absolute path of validation data<br />
&nbsp;&nbsp;&nbsp;&nbsp;training_patch: absolute path of training_patch<br />
&nbsp;&nbsp;&nbsp;&nbsp;testing_patch: absolute path of testing_patch<br />

# Commands to set up in conda:

conda create -n LDAMP_tf python=3.7.9<br />
conda activate LDAMP_tf<br />
pip install tensorflow-gpu==1.15<br />
pip install matplotlib<br />


# Data:

TrainingData_patch40.npy # found on the original github google drive<br />
StandardTestData_256Res.npy<br />
