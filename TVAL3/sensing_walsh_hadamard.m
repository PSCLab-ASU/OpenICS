function [A, At] = sensing_walsh_hadamard(n,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    
    % generate measurement matrix
    p = randperm(n);
    picks = p(1:m);
    for ii = 1:m
        if picks(ii) == 1
            picks(ii) = p(m+1);
            break;
        end
    end
    perm = randperm(n); % column permutations allowable
    
    A = @(z) A_fWH(reshape(z, [], 1), picks, perm);
    At = @(z) At_fWH(reshape(z, [], 1), picks, perm);
end
