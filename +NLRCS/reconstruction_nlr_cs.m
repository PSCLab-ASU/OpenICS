% reconstruction_tv.m
% 
% Uses NLR-CS method from original source code.
% 
% Usage: x_hat = reconstruction_nlr_cs(x,y,img_dims,A,At,specifics)
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

function x_hat = reconstruction_nlr_cs(x, y, img_dims, A, At, specifics)

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
    % note, NLR-CS uses 0-255 instead of 0-1, so multiply image data by 255
    specifics.y = A(x .* 255);
    specifics.ori_im = x .* 255;
    specifics.s_model = 1;
    q = randperm(round(numel(x)/2)-1)+1;
    specifics.picks = q(1:numel(y) / 2)';

    time0 = clock;
    % divide image data by 255 to get back to 0-1 range
    x_hat = NLR_CS_Reconstruction(specifics, A, At) ./ 255;
    runtime = etime(clock, time0);
    fprintf('Total elapsed time = %f secs\n\n', runtime);
end
