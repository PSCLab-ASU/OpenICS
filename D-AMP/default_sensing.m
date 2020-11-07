function sensing = default_sensing(reconstruction_method)
% returns the name of the preferred sensing method for the specified
% reconstruction method

    switch reconstruction_method
        case 'reconstruction_damp'
            sensing = 'sensing_guassian_random_columnwise';
        otherwise
            sensing = 'sensing_guassian_random';
    end

end
