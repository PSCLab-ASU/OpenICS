% reconstruction_damp.m
% 
% Uses D-AMP reconstruction method.
% 
% Usage: x_hat = reconstruction_damp(x,y,input_channel,input_width,input_height,A,At,specifics)
%
% x - nx1 vector, original signal
%
% y - mx1 vector, observations
%
% input_channel - Channels in the original image
%
% input_width - Width of the original image
%
% input_height - Height of the original image
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

function x_hat = reconstruction_damp(x, y, input_channel, input_width, input_height, A, At, specifics)

    % set default values
    if ~isfield(specifics, 'denoiser')
        specifics.denoiser = 'BM3D';
    end
    
    if ~isfield(specifics, 'iters')
        specifics.denoiser = 30;
    end

    time0 = clock;
    [x_hat, PSNR] = DAMP(y, specifics.iters, input_width, input_height, specifics.denoiser, A, At);
    fprintf('Total elapsed time = %f secs\n\n', etime(clock,time0));
end
