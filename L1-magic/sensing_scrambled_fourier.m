function [A, At] = sensing_scrambled_fourier(n,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % used for smaller scale images
    
    % permutation P and observation set OMEGA
    P = randperm(n)';
    q = randperm(n/2-1)+1;
    OMEGA = q(1:m/2)';
    
    % function handles to A_f and At_f functions
    A = @(z) A_f(z, OMEGA, P);
    At = @(z) At_f(z, n, OMEGA, P);
end
