function sensing = default_sensing(reconstruction_method)
% returns the name of the preferred sensing method for the specified
% reconstruction method

    switch reconstruction_method
        case 'reconstruction_nlr_cs'
            sensing = 'sensing_scrambled_fourier';
        otherwise
            sensing = 'sensing_guassian_random';
    end

end
