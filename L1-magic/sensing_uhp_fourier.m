function [A, At] = sensing_uhp_fourier(c,w,h,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % used more often for largescale images
    % IMPORTANT: this method requires n to have an integer square root
    
    % creates random observation set OMEGA for Fourier coefficients
    n = c * w * h;
    q = randperm(n/2-1)+1;
    OMEGA = q(1:m/2)';
    n = sqrt(n);
    size(OMEGA)
    
    % check square root is an integer
    if mod(n, 1) ~= 0
        error('ERROR: Image does not have square dimensions');
    end
    
    % function handles to A_fhp and At_fhp functions
    A = @(z) A_fhp(reshape(z, [], 1), OMEGA);
    At = @(z) At_fhp(reshape(z, [], 1), OMEGA, n);
end
