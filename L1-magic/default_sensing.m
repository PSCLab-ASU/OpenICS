function sensing = default_sensing(reconstruction_method)
% returns the name of the preferred sensing method for the specified
% reconstruction method

    switch reconstruction_method
        case 'reconstruction_tv'
            sensing = 'sensing_scrambled_fourier';
        case 'reconstruction_l1'
            sensing = 'sensing_uhp_fourier';
        otherwise
            sensing = 'sensing_guassian_random';
    end

end
