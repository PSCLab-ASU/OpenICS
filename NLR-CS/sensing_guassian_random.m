function [A, At] = sensing_guassian_random(c,w,h,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % WARNING: significantly slower on large images due to sensing matrix
    % being stored in memory
    
    n = w * h * c;
    
    if n > 10000
        disp('WARNING: Explicit matrix sensing is significantly slower for larger images!');
    end
    
    Amat = normrnd(0, 1/m, m, n); % random sensing matrix
    
    %Amat = normrnd(0, 1/m, n, n);
    %Amat = qr(Amat')';
    %Amat = Amat(1:m,:);
    
    A = @(z) Amat * z(:);
    At = @(z) Amat' * z(:);
end
