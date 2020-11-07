function [A, At] = sensing_scrambled_fourier(img_dims,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    
    % permutation P and observation set OMEGA
    n = prod(img_dims);
    P = randperm(n)';
    q = randperm(n/2-1)+1;
    OMEGA = q(1:m/2)';
    
    % function handles to A_f and At_f functions
    A = @(z) NLR_A_f(z(:), OMEGA, P);
    At = @(z) NLR_At_f(z(:), n, OMEGA, P);
end
