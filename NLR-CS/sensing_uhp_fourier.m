function [A, At] = sensing_uhp_fourier(img_dims,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % IMPORTANT: this method requires n to have an integer square root
    
    % creates random observation set OMEGA for Fourier coefficients
    n = prod(img_dims);
    q = randperm(n/2-1)+1;
    OMEGA = q(1:m/2)';
    
    % function handles to A_fhp and At_fhp functions
    A = @(z) NLR_A_fhp(reshape(z, [], 1), OMEGA, img_dims(3), img_dims(2));
    At = @(z) NLR_At_fhp(reshape(z, [], 1), OMEGA, img_dims(3), img_dims(2));
end
