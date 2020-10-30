% reconstruction_tv.m
% 
% Uses NLR-CS method from original source code.
% 
% Usage: x_hat = reconstruction_nlr_cs(x,y,input_channel,input_width,input_height,A,At,specifics)
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
% specifics - struct that may contain the following arguments:
%
%   win - The size of each patch.
%       Default: 6
%
%   nblk - The number of similar patches in each grouping.
%       Default: 45
%
%   step - The steps between patches.
%       Default: 5
%
%   K0 - Iterations of NLR-CS after which to apply adaptive weights to SVT.
%       Default: 3
%
%   K - Iterations of NLR-CS to run.
%       Default: 18
%
%   t0 - Threshold for DCT-thresholding.
%       Default: Varies based on sampling ratio
%
%   nSig - Threshold modifier for DCT-thresholding.
%       Default: Varies based on sampling ratio
%
%   c0 - Threshold for non-weighted SVT.
%       Default: Varies based on sampling ratio
%
%   c1 - Threshold for weighted SVT.
%       Default: Varies based on sampling ratio
%

function x_hat = reconstruction_nlr_cs(x, y, input_channel, input_width, input_height, A, At, specifics)

    % set default values based on Set_parameters method
    defaults = Set_parameters(0.1, 0, 1);
    
    f = fieldnames(defaults);
    for i = 1:length(f)
        
        % if field does not exist, set to default value
        if ~isfield(specifics, f{i})
            specifics.(f{i}) = defaults.(f{i});
        end
        
    end
    
    % set code-defined properties
    specifics.y = A(x .* 255);
    specifics.ori_im = x .* 255;
    specifics.s_model = 1;
    q = randperm(numel(x)/2-1)+1;
    specifics.picks = q(1:numel(y) / 2)';

    time0 = clock;
    x_hat = NLR_CS_Reconstruction(specifics, A, At) ./ 255;
    fprintf('Total elapsed time = %f secs\n\n', etime(clock,time0));
end
