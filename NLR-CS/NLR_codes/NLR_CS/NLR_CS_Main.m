% =========================================================================
% NLR Compressive Sensing, Version 1.0
% Copyright(c) 2013 Weisheng Dong
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for NLR-CS
% 
% Please refer to the following paper if you use this code:
% W. Dong, G. Shi, X. Li, Y. Ma, and F. Huang, "Compressive sensing via 
% Nonlocal Low-rank Regularization," submitted to IEEE Trans. on Image
% Processing, 2014.
% 
%--------------------------------------------------------------------------
clc;
clear;
addpath('Utilities');
addpath('Utilities/Measurements');

Test_image_dir     =    'Data/CS_test_images';
Image_name         =    'Barbara.tif';

rates              =    [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
L                  =    [20, 35, 50, 65, 80, 95];

s_model            =    1;   %   1: Random sampling used in L1-magic (subsampling in 1D FFT domain);       2: Pseudo radial line sampling
method             =    1;   %   1: NLR-baseline method;                                                   2: NLR method
idx                =    2; 

ori_im             =    double( imread( fullfile(Test_image_dir, Image_name)) );
Rec_im             =    Image_CS( ori_im, s_model, rates(idx), L(idx) );

imwrite( Rec_im./255, 'Results/Rec_Barbara.tif' );