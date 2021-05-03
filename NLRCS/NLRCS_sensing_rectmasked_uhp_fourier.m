function [A, At] = sensing_rectmasked_uhp_fourier(img_dims,m)
    % returns two function handles:
    % one for regular sensing and one for transposed sensing
    
    % creates random rectangular mask for upper half-plane Fourier coefficients
    rate = m / prod(img_dims);
    
    if rate==0.2
        factor = 4.427;
    elseif rate==0.25
        factor = 4;
    else
        factor = sqrt(1/rate)*2;
    end
    
    w = img_dims(2);
    h = img_dims(3);
    picks = RandMask_rect(double(h/factor), double(w/factor), h, w);
    
    % function handles to A_fhp and At_fhp functions
    A = @(z) NLR_A_fhp(z(:), picks, h, w);
    At = @(z) NLR_At_fhp(z(:), picks, h, w);
end
