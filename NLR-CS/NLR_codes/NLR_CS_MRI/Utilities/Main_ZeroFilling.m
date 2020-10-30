% =========================================================================
% LASSC Compressive Sensing, Version 1.0
% Copyright(c) 2011 Weisheng Dong
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
% ----------------------------------------------------------------------
% 
% This is an implementation of the algorithm for LASSC-CS
% 
% Please refer to the following paper if you use this code:
% 
% 
% -----------------------------------------------------------------------
clear; %close all;
addpath('Utilities');
addpath('Utilities\Measurements');
addpath('Data');
addpath('Results');

Test_data_No    =   3;   %  1:RealMRI;  2:Brain512;   3:Cart

switch Test_data_No
    case {1}
        Test_data    =   'Data\MRI_33.mat';
        Result_dir   =   'Results\Real';
        fn           =   'Real_';
    case {2}
        Test_data    =  'Data\Brain512.mat';
        Result_dir   =   'Results\Brain512';
        fn           =   'Brain512_';
    case {3}
        Test_data    =  'Data\k-space.mat';
        Result_dir   =   'Results\Cart';
        fn           =   'Cart_';
end
load( Test_data );

switch Test_data_No
    case {1}
        I2                =   zeros(256,256);
        I2(9:248,9:248)   =   I;
        I                 =   I2;
    case {2}
        I          =   fftshift( ifft2( fftshift(data)) );
    case {3}
        I          =   squeeze( data(:,:,8,2) );
        I          =   fftshift( fft2( fftshift(I) ) );
end
        
rates         =    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1, 1, 1, 1, 1, 1];
L             =    [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80];

Mode          =   3;
pre           =   'ZF_m3_';
cnt           =   0;
sum_psnr      =   0;
sum_snr       =   0;
time0         =   clock;
fn_txt        =   strcat( pre, 'PSNR_SNR.txt' ); 
fd_txt        =   fopen( fullfile(Result_dir, fn_txt), 'wt');

for i = 5 : length( L )
    
    [h, w]       =   size( I );
    I            =   I./max( abs(I(:)) );
    
    [im_res, PSNR, SNR]    =   ZF_CS_MRI( I, rates(i), L(i), Mode);
    
    sum_psnr           =   sum_psnr + PSNR;
    sum_snr            =   sum_snr + SNR;    
    fname              =   [pre, fn, num2str(L(i)), '.png'];
        
    imwrite(abs(im_res), fullfile(Result_dir, fname));    
    disp( sprintf('%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, L(i), PSNR, SNR) );
    fprintf(fd_txt, '%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, L(i), PSNR, SNR);
    cnt   =  cnt + 1;    
end
fprintf(fd_txt, '\n\nAverage :  PSNR = %2.2f  SSIM = %2.4f\n', sum_psnr/cnt, sum_snr/cnt);
fclose(fd_txt);
disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));


% % ************************************************************************/
% % Exp 2
% % ************************************************************************/
% clear; %close all;
% addpath('Utilities');
% addpath('Utilities\Measurements');
% addpath('Data');
% addpath('Results');
% 
% Test_data_No    =   2;   %  1:RealMRI;  2:Brain512;   3:Cart
% 
% switch Test_data_No
%     case {1}
%         Test_data    =   'Data\MRI_33.mat';
%         Result_dir   =   'Results\Real';
%         fn           =   'Real_';
%     case {2}
%         Test_data    =  'Data\Brain512.mat';
%         Result_dir   =   'Results\Brain512';
%         fn           =   'Brain512_';
%     case {3}
%         Test_data    =  'Data\k-space.mat';
%         Result_dir   =   'Results\Cart';
%         fn           =   'Cart_';
% end
% load( Test_data );
% 
% switch Test_data_No
%     case {1}
%         I2                =   zeros(256,256);
%         I2(9:248,9:248)   =   I;
%         I                 =   I2;
%     case {2}
%         I          =   fftshift( ifft2( fftshift(data)) );
%     case {3}
%         I          =   squeeze( data(:,:,8,2) );
%         I          =   fftshift( fft2( fftshift(I) ) );
% end
%         
% rates         =    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
% L             =    [10, 15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 70, 80];
% 
% Mode          =   2;
% pre           =   'LRCS_m2_';
% cnt           =   0;
% sum_psnr      =   0;
% sum_snr       =   0;
% time0         =   clock;
% fn_txt        =   strcat( pre, 'PSNR_SNR.txt' ); 
% fd_txt        =   fopen( fullfile(Result_dir, fn_txt), 'wt');
% 
% for i = 1 : length( rates )
%     
%     [h, w]       =   size( I );
%     I            =   I./max( abs(I(:)) );
%     
%     [im_res, PSNR, SNR]    =   CS_MRI( I, rates(i), L(i), Mode);
%     
%     sum_psnr           =   sum_psnr + PSNR;
%     sum_snr            =   sum_snr + SNR;    
%     fname              =   [pre, fn, num2str(rates(i)), '.png'];
%         
%     imwrite(abs(im_res), fullfile(Result_dir, fname));    
%     disp( sprintf('%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, rates(i), PSNR, SNR) );
%     fprintf(fd_txt, '%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, rates(i), PSNR, SNR);
%     cnt   =  cnt + 1;    
% end
% fprintf(fd_txt, '\n\nAverage :  PSNR = %2.2f  SSIM = %2.4f\n', sum_psnr/cnt, sum_snr/cnt);
% fclose(fd_txt);
% disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));



% ************************************************************************/
% Exp 3
% ************************************************************************/
clear; %close all;
addpath('Utilities');
addpath('Utilities\Measurements');
addpath('Data');
addpath('Results');

Test_data_No    =   3;   %  1:RealMRI;  2:Brain512;   3:Cart

switch Test_data_No
    case {1}
        Test_data    =   'Data\MRI_33.mat';
        Result_dir   =   'Results\Real';
        fn           =   'Real_';
    case {2}
        Test_data    =  'Data\Brain512.mat';
        Result_dir   =   'Results\Brain512';
        fn           =   'Brain512_';
    case {3}
        Test_data    =  'Data\k-space.mat';
        Result_dir   =   'Results\Cart';
        fn           =   'Cart_';
end
load( Test_data );

switch Test_data_No
    case {1}
        I2                =   zeros(256,256);
        I2(9:248,9:248)   =   I;
        I                 =   I2;
    case {2}
        I          =   fftshift( ifft2( fftshift(data)) );
    case {3}
        I          =   squeeze( data(:,:,8,2) );
        I          =   fftshift( fft2( fftshift(I) ) );
end
        
rates         =    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
L             =    [10, 15, 20, 25, 35, 40, 45, 50, 55, 60, 65, 70, 80];

Mode          =   3;
pre           =   'LRCS_m3_';
cnt           =   0;
sum_psnr      =   0;
sum_snr       =   0;
time0         =   clock;
fn_txt        =   strcat( pre, 'PSNR_SNR.txt' ); 
fd_txt        =   fopen( fullfile(Result_dir, fn_txt), 'wt');

for i = 6 : length( rates )
    
    [h, w]       =   size( I );
    I            =   I./max( abs(I(:)) );
    
    [im_res, PSNR, SNR]    =   CS_MRI( I, rates(i), L(i), Mode);
    
    sum_psnr           =   sum_psnr + PSNR;
    sum_snr            =   sum_snr + SNR;    
    fname              =   [pre, fn, num2str(rates(i)), '.png'];
        
    imwrite(abs(im_res), fullfile(Result_dir, fname));    
    disp( sprintf('%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, L(i), PSNR, SNR) );
    fprintf(fd_txt, '%s, rate %2.2f: PSNR = %3.2f  SSIM = %f\n', Test_data, L(i), PSNR, SNR);
    cnt   =  cnt + 1;    
end
fprintf(fd_txt, '\n\nAverage :  PSNR = %2.2f  SSIM = %2.4f\n', sum_psnr/cnt, sum_snr/cnt);
fclose(fd_txt);
disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));

