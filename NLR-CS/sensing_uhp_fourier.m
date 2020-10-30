function [A, At] = sensing_uhp_fourier(c,w,h,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    % used more often for largescale images
    % IMPORTANT: this method requires n to have an integer square root
    
    % creates random rectangular mask for Fourier coefficients
    rate = m / w / h / c;
    
    if rate==0.2
        factor = 4.427;
    elseif rate==0.25
        factor = 4;
    else
        factor = sqrt(1/rate)*2;
    end
    
    picks = RandMask_rect(double(h/factor), double(w/factor), h / 4, w / 4);
    
    % function handles to A_fhp and At_fhp functions
    A = @(z) A_fhp(z(:), picks, h, w);
    At = @(z) At_fhp(z(:), picks, h, w);
end
