# D-AMP Toolbox

Matlab and TensorFlow implementations of algorithms from the following papers.

Metzler, Christopher A., Arian Maleki, and Richard G. Baraniuk. "From denoising to compressed sensing." IEEE Transactions on Information Theory 62.9 (2016). http://arxiv.org/abs/1406.4175

Schniter, Philip, Sundeep Rangan, and Alyson Fletcher. "Denoising based vector approximate message passing." arXiv preprint arXiv:1611.01376 (2016). http://arxiv.org/abs/1611.01376

Metzler, Christopher A., Arian Maleki, and Richard G. Baraniuk. "BM3D-prgamp: compressive phase retrieval based on BM3D denoising." Image Processing (ICIP), 2016 IEEE International Conference on. IEEE, 2016. http://ieeexplore.ieee.org/abstract/document/7532810/

Metzler, Christopher A., Ali Mousavi, and Richard Baraniuk. "Learned D-AMP: Principled Neural Network based Compressive Image Recovery." Advances in Neural Information Processing Systems. 2017. http://papers.nips.cc/paper/6774-learned-d-amp-principled-neural-network-based-compressive-image-recovery.pdf

Metzler, Christopher A., Ali Mousavi, Reinhard Heckel, and Richard Baraniuk. "Unsupervised Learning with Stein's Unbiased Risk Estimator." https://arxiv.org/abs/1805.10531

Questions/suggestions/comments: chris.metzler@rice.edu (D-AMP, D-prGAMP, LDAMP, and SURE) or schniter.1@osu.edu (D-VAMP)

---------------------------------------------------------------------------

# Primary Contents
## Scripts:
`CS_1D_Demo.m`: Recover a compressively sampled 1D signal with Haar wavelet sparsity based (V)AMP and NLM-(V)AMP.

`CS_Imaging_Demo.m`: Recover compressively sampled image using D-AMP.

`CS_Imaging_Demo_DVAMP.m`: Recover compressively sampled image using D-VAMP.

`CS_Imaging_Demo_LDAMP.m`: Recover compressively sampled image using L(V)AMP.

`CPR_Imaging_Demo.m`: Perform compressive phase retrieval with D-prGAMP.

`QQplotGenerator.m`: Generate a series of QQplots of the effective noise of D-IT and D-AMP.

`StateEvolutionGenerator.m`: Compute the state evolution of a D-AMP algorithm and compare it to the true MSEs of D-AMP and D-IT.

## Functions:
`AMP.m`: Reconstructs sparse, compressively sampled signals using AMP.

`VAMP.m`: Reconstructs sparse, compressively sampled signals using VAMP.

`DAMP.m`: Performs D-AMP reconstruction of a compressively sampled signal. The string "denoiser" selects which denoiser to use.

`DVAMP.m`: Performs D-VAMP reconstruction of a compressively sampled signal. The string "denoiser" selects which denoiser to use.

`DprGAMP.m`: Performs D-prGAMP compressive phase retrieval of a signal. The string "denoiser" selects which denoiser to use.

`DIT.m`: Performs D-IT reconstruction of a compressively sampled signal. The string "denoiser" selects which denoiser to use.

`DAMP_oneIter.m`: Performs a single iteration of D-AMP.  Used to generate state evolution and qqplots.

`DIT_oneIter.m`: Performs a single iteration of D-IT.  Used to generate state evolution and qqplots.

`DAMP_SE_Prediction.m`: Computes the next predicted state evolution of a D-AMP algorithm.

`denoise.m`: Denoises an incoming signal.  Which denoiser is used depends on the string "denoiser".  Currently supports Gaussian filtering, bilateral filtering, NLM, BLS-GSM, BM3D, BM3D-SAPCA, and DnCNN.  Add your own denoising algorithms here.

## Auxiliary functions:
`phi_fp.m`: Projects a length n signal to m measurements using a Gaussian random matrix. Can handle very large measurement matrices but is very slow.

`phit_fp.m`: Projects a length m signal onto the transpose of an mxn Gaussian measurement matrix. Can handle very large measurement matrices but is very slow.

`psi_fp.m`: Transforms wavelet coefficients into the pixel domain.

`psit_fp.m`: Transforms pixels to their wavelet coefficients.

`PSNR.m`: Computes the PSNR of an image x and its estimate x_hat.

`LoadNetworkWeights.m`: Load the DnCNN denoisers' network weights.

## Other files:
`OptimumLambdaSigned.mat`: Look up table to set the threshold for AMP.

---------------------------------------------------------------------------  
  
# LDAMP_TensorFlow Contents
## Code:
`TrainLearnedDAMP.py`: Code to train the LDAMP or LDIT networks. Supports layer-by-layer and end-to-end training. Supports training with SURE and GSURE loss.

`TestLearnedDAMP.py`: Code to test the LDAMP or LDIT networks. Supports layer-by-layer end-to-end, and denoiser-by-denoiser trained networks.

`TrainDnCNN.py`: Code to train a TensorFlow implementation of DnCNN. Supports training with SURE loss.

`TestLearnedDAMP.py`: Code to test a TensorFlow implementation of DnCNN.

`LearnedDAMP.py`: Module that contains the implementations of and support functions for LDAMP, LDIT, and DnCNN.

`Training_Script.sh`: Bash script to train LDAMP, LDIT, and DnCNN networks.

## Data:
saved_models and TrainingData are both empty directories on github. Their contents can be downloaded from https://rice.app.box.com/v/LDAMP-LargeFiles. You may run into Box's file size limit. In that case download one directory at a time.

**Important**: At present the TensorFlow networks are smaller than the Matlab networks and have been trained with less data. For the best possible LDAMP performance use the 20 layer version of the "DnCNN" denoiser in CS_Imaging_Demo_LDAMP.m

---------------------------------------------------------------------------
# SURE_deep_image_prior Contents
`denoising_wSURE.py`: Code to denoise an image using a randomly initialized CNN using SURE.

Other files are dependencies and come from https://github.com/DmitryUlyanov/deep-image-prior.

---------------------------------------------------------------------------
# Packages
This download includes the BM3D, BLS-GSM, NLM, and Rice Wavelet Toolbox packages.
The latest versions of these packages can be found at:

BM3D: http://www.cs.tut.fi/~foi/GCF-BM3D/

BLS-GSM: http://decsai.ugr.es/~javier/denoise/software/index.htm

NLM: http://www.mathworks.com/matlabcentral/fileexchange/27395-fast-non-local-means-1d--2d-color-and-3d

Rice Wavelet Toolbox (RWT): https://github.com/ricedsp/rwt

# Dependencies
The TensorFlow LDAMP/LDIT demos require that models and training data be downloaded from https://rice.app.box.com/v/LDAMP-LargeFiles. Place the models and data in D-AMP_Toolbox/LDAMP_TensorFlow/saved_models/ and D-AMP_Toolbox/LDAMP_TensorFlow/TrainingData/ respectively.

The Matlab LDAMP/LDVAMP demos (DnCNN-AMP) require that you have Matconvnet (http://www.vlfeat.org/matconvnet/) compiled and on your path.

VAMP demos require that you have the latest version of the GAMPmatlab toolbox downloaded via the SVN interface (svn co svn://svn.code.sf.net/p/gampmatlab/code/ GAMPmatlab) and that you have GAMPmatlab/main and GAMPmatlab/VAMP on your path.

RED demos require that you have FASTA on your path (https://github.com/tomgoldstein/fasta-matlab).

# Installation
The LDAMP_TensorFlow code were created and tested using TensorFlow 1.7 and Python 2.7.

The Non-local Means and Rice Wavelet Toolbox code need to be compiled before they can be used:

Compile the NLM utilities by typing `mex image2vectors_double.c`, `mex image2vectors_single.c`, `mex vectors_nlmeans_double.c`, and `mex vectors_nlmeans_single.c` from the Packages/NLM/ directory. 

Compile the Rice Wavelet Toolbox (RWT) by entering `compile` from the Packages/rwt/bin/ directory. Currently the RWT does not compile under Windows 64 bit Matlab.

You will need Matlab's stat toolbox for qqplots.


# Example
Use BM3D-AMP to recover a signal a 128x128 image x_0 sampled according to y=M*x_0 where M is an m by n Gaussian measurement matrix with unit norm columns:
    `x_hat=DAMP(y,30,128,128,'BM3D',M);`
See CS_Imaging_Demo.m for other examples, including D-VAMP.


# Modifying Code
Our code was designed to make it as easy as possible to test D-(V)AMP with a new denoiser.
To test another denoiser in the D-(V)AMP algorithm, simply add an additional case statement to denoise.m and place your denoising function there.  (Your denoiser will have access to the noisy signal as well as an estimate of the standard deviation of the noise).
Next change the "denoiser" argument in any calls to DAMP, DVAMP, DprGAMP, DIT, DAMP_oneIter, etc., to the name used in your case statement. e.g: `x_hat=DAMP(y,30,128,128,'mydenoiser',M);`

# Known Issues
The latest version of the Rice Wavelet Toolbox (RWT) does not compile under Windows 64 bit Matlab. Older versions of the RWT often crash. 
At this point in time functions and scripts which use the RWT (CS_1D_Demo and the BLS-GSM denoiser) must be run on a Unix machine.  BM3D-AMP and other D-AMP algorithms work with Windows and Unix.

# Acknowledgements
Thanks to David Van Veen for help implementing an rgb version of D-AMP.

### Original release date : 8/7/14
