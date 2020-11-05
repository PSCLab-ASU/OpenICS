Variables:

sensing: unused
reconstruction: type of reconstruction (LDAMP)
stage: either "training" or "testing"
default: default is training
dataset: name of dataset to train on
input_channel: number of channels the training data has
input_width: width of training data images
input_height: height of training data images
m: manually calculated
n: manually calculated
specifics: (used to store variables conveniently and define new ones)
	channel_img: stored from original input
	width_img: stored from original input
	height_img: stored from original input
	n: manually calculated
	m: manually calculated
	sampling_rate: rate at which the training data is sampled
	alg: algorithm used, in this case DAMP
	tie_weights: if true, only one layer of LDAMP is trained
	filter_height: denoiser filter height
	filter_width: denoiser filter width
	num_filters: number of filters in each layer of the DnCNN
	n_DnCNN_layers: number of denoiser layers
	max_n_DAMP_layers: number of LDAMP layers to train to or layer to test at
	start_layer: which layer to start on
	max_Epoch_Fails: How many training epochs to run without improvement in the validation error
	ResumeTraining: option to resume training a network you were previously training
	LayerByLayer: true if training layer by layer
	DenoiserbyDenoiser: true if training denoiser by denoiser (overrides LayerByLayer)
	sigma_w_min: used in denoiesrbydenoiser training, determines noise
	sigma_w_max: used in denoiesrbydenoiser training, determines noise
	sigma_w: used in both LbL and DbD to determine noise (LbL testing: 0, LbL training: 1./255., DbD test and train: 25./255.)
	learning_rates: learning rates used, once one does not yield improvement -> move to next
	EPOCHS: number of epochs
	n_Train_Images: number of training images used
	n_Val_Images: number of validation images used
	BATCH_SIZE: batch size used during training and testing (typically 128 for training, 1 for testing)
	InitWeightsMethod: way to initialize the weights, (default: smaller_net; options: random, denoiser, smaller net, and layer_by_layer)
	loss_func: loss function used, can only use MSE currently
	init_mu: mean when initializing denoiser layers
	init_sigma: sigma (std deviation) when initializing denoiser layers
	mode: sensing mode (default: gaussian)
	validation_patch: absolute path of validation data
	training_patch: absolute path of training_patch
	testing_patch: absolute path of testing_patch

Commands to set up in conda:

conda create -n LDAMP_tf python=3.7.9
conda activate LDAMP_tf
pip install tensorflow-gpu==1.15
pip install matplotlib


Data:

TrainingData_patch40.npy # found on the original github google drive
StandardTestData_256Res.npy
