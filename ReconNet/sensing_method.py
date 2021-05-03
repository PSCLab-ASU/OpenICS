import ReconNet.utils as u

def sensing_method(method_name, n, m, input_width, input_height, input_channel):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    if method_name.lower() == 'random':
        print("Using random sensing")
        return u.random_sensing(m,input_width,input_height,input_channel)
    else:
        raise NotImplementedError
