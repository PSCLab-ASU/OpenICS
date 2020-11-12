% reconstruction_damp.m
% 
% Uses D-AMP reconstruction method.
% 
% Usage: x_hat = reconstruction_damp(x,y,img_dims,A,At,specifics)
%
% x - nx1 vector, original signal
%
% y - mx1 vector, observations
%
% img_dims - Vector denoting the size of the original image. [c,w,h]
%
% A - Function handle to sensing method
%
% At - Function handle to transposed sensing method
%
% specifics - Map that must contain at least the following arguments:
%
%   denoiser - The denoiser to use. Currently includes the following:
%              'NLM', 'Gauss', 'Bilateral', 'BLS-GSM', 'BM3D', 'fast-BM3D',
%              and 'CBM3D'.
%       Default: 'BM3D'
%
%   iters - How many iterations of D-AMP to run.
%       Default: 30
%

function x_hat = reconstruction_damp(x, y, img_dims, A, At, specifics)

    % set default values
    if ~isfield(specifics, 'denoiser')
        specifics.denoiser = 'BM3D';
    end
    
    if ~isfield(specifics, 'iters')
        specifics.iters = 30;
    end
    
    % D-AMP operates on 0-255, so modify y
    y = y .* 255;

    time0 = clock;
    [x_hat, PSNR] = DAMP(y, specifics.iters, img_dims(2), img_dims(3), specifics.denoiser, A, At);
    fprintf('Total elapsed time = %f secs\n\n', etime(clock,time0));
    
    % divide by 255 to shift back to 0-1 range
    x_hat = x_hat ./ 255;
end
