function [A, At] = sensing_guassian_random(n,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % WARNING: significantly slower on large images due to sensing matrix
    % being stored in memory
    
    if n > 10000
        disp('WARNING: Guassian sensing is significantly slower for larger images!');
    end
    
    Amat = normrnd(0, 1/m, m, n); % random sensing matrix
    A = @(z) Amat * reshape(z, [], 1);
    At = @(z) Amat' * reshape(z, [], 1);
end