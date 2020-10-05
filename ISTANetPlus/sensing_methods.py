import scipy.io as sio
import numpy as np
import os

def sensing_method(method_name,specifics):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    return 1

def computInitMx(Training_labels, specifics):
    Phi_data_Name = './%s/phi_0_%d_1089.mat' % (specifics['matrix_dir'], specifics['cs_ratio'])
    Phi_data = sio.loadmat(Phi_data_Name)
    Phi_input = Phi_data['phi']

    Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (specifics['matrix_dir'], specifics['cs_ratio'])

    # Computing Initialization Matrix:
    if os.path.exists(Qinit_Name):
        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']

    else:
        X_data = Training_labels.transpose()
        Y_data = np.dot(Phi_input, X_data)
        Y_YT = np.dot(Y_data, Y_data.transpose())
        X_YT = np.dot(X_data, Y_data.transpose())
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
        del X_data, Y_data, X_YT, Y_YT
        sio.savemat(Qinit_Name, {'Qinit': Qinit})
    return Phi_input, Qinit