function [A, At] = sensing_guassian_random_columnwise(n,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % WARNING: significantly slower on large images due to sensing matrix
    % being stored in memory
    
    if n > 10000
        disp('WARNING: Explicit matrix sensing is significantly slower for larger images!');
    end
    
    M=randn(m,n);
    for j = 1:n
        M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
    end
    
    A = @(z) M * reshape(z, [], 1);
    At = @(z) M' * reshape(z, [], 1);
end
