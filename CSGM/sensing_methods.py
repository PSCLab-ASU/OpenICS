
import numpy as np
def sensing_method(method_name,n,m,imgdim,channel,path = None):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    if method_name == 'gaussian':
        sensing_method = lambda: np.random.randn(n, m)
    if method_name == 'gaussian_fixed':
        matrix = np.random.randn(n, m)
        sensing_method = lambda: matrix
    return sensing_method

