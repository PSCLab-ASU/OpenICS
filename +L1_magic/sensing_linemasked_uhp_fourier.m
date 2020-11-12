function [A, At] = sensing_linemasked_uhp_fourier(img_dims,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % IMPORTANT: this method requires n to have an integer square root
    
    % creates line mask observation set OMEGA for Fourier coefficients
    n = round(sqrt(prod(img_dims)));
    [M,Mh,mh,mhi] = LineMask(m,n);
    OMEGA = mhi;
    
    % function handles to L1_A_fhp and L1_At_fhp functions
    A = @(z) L1_A_fhp(reshape(z, [], 1), OMEGA);
    At = @(z) L1_At_fhp(reshape(z, [], 1), OMEGA, n);
end
