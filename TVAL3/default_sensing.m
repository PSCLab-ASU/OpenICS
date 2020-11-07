function sensing = default_sensing(reconstruction_method)
% returns the name of the preferred sensing method for the specified
% reconstruction method

    switch reconstruction_method
        case 'reconstruction_tval3'
            sensing = 'sensing_walsh_hadamard';
        otherwise
            sensing = 'sensing_guassian_random';
    end

end
