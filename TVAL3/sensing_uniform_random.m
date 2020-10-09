function [A, At] = sensing_uniform_random(n,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % WARNING: significantly slower on large images due to sensing matrix
    % being stored in memory
    
    if n > 10000
        disp('WARNING: Guassian sensing is significantly slower for larger images!');
    end
    
    Amat = rand(m, n) - 0.5; % random sensing matrix
    A = @(z) Amat * reshape(z, [], 1);
    At = @(z) Amat' * reshape(z, [], 1);
end
