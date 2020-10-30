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
clear
clear;
addpath('Utilities');
addpath('Utilities\Measurements');
addpath('Data');

Test_MRI_data     =   1;

switch Test_MRI_data
    case {1}
        Data_file    =   'Data\Ori_Head_MRI.mat';
        Result_dir   =   'Results\Head_results';
        fn           =   'Head_';
    case {2}
        Data_file    =   'Data\Ori_Brain_MRI.mat';
        Result_dir   =   'Results\Brain_results';
        fn           =   'Brain_';
end
load( Data_file );
        
rates         =    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
L             =    [15, 25, 35, 45, 55, 65, 80];
idx           =    3;
s_Mode        =    1;           %  1 : 2D Random subsampling;  2 : Pseudo radial lines subsampling
pre           =    'NLR_';

[h, w]        =    size( I );
I             =    I./max( abs(I(:)) );

    
[Rec_im, PSNR]     =   CS_MRI( I, rates(idx), L(idx), s_Mode);

disp( sprintf('NLR-CS-MRI reconstruction: PSNR = %3.2f\n', PSNR) );

Fname              =   ['NLR_', fn, num2str(idx), '.png'];
imwrite(abs(Rec_im), fullfile(Result_dir, Fname));    

Fname              =   ['NLR_', fn, num2str(idx), '.mat'];
save(fullfile(Result_dir, Fname), 'Rec_im');


