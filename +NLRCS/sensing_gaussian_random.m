function [A, At] = sensing_gaussian_random(img_dims,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % WARNING: significantly slower on large images due to sensing matrix
    % being stored in memory
    
    n = prod(img_dims);
    
    if n > 10000
        disp('WARNING: Explicit matrix sensing is significantly slower for larger images!');
    end
    
    % random orthogonalized sensing matrix
    Amat = randn(m, n);
    Amat = orth(Amat')';
    
    A = @(z) Amat * reshape(z, [], 1);
    At = @(z) Amat' * reshape(z, [], 1);
end
