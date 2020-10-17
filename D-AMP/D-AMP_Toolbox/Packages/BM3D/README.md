# MATLAB wrapper for BM3D denoising - from Tampere with love

MATLAB wrapper for BM3D for stationary correlated noise (including white noise) for color,
grayscale and multichannel images and deblurring.

BM3D is an algorithm for attenuation of additive spatially correlated
stationary (aka colored) Gaussian noise. This package provides a wrapper
for the BM3D binaries for use for grayscale, color and other multichannel images
for denoising and deblurring.

This implementation is based on Y. Mäkinen, L. Azzari, A. Foi,
"Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise",
in Proc. 2019 IEEE Int. Conf. Image Process. (ICIP), pp. 185-189.

The package contains the BM3D binaries compiled for:
- Windows (R2018b)
- Linux (R2017b)
- Mac OSX (R2018a)

The binaries are available for non-commercial use only. For details, see LICENSE.

Authors:
	Ymir Mäkinen   <ymir.makinen@tuni.fi>
	Lucio Azzari
	Alessandro Foi



