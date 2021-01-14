#TODO tensorflow version 2.X migration code changed the import tensorflow as tf line to two lines as seen below
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def SetNetworkParams(new_height_img, new_width_img,new_channel_img, new_filter_height,new_filter_width,\
                     new_num_filters,new_n_DnCNN_layers,new_n_DAMP_layers, new_sampling_rate,\
                     new_BATCH_SIZE,new_sigma_w,new_n,new_m,new_training, iscomplex, use_adaptive_weights=False):
    global height_img, width_img, channel_img, filter_height, filter_width, num_filters, n_DnCNN_layers, n_DAMP_layers,\
        sampling_rate, BATCH_SIZE, sigma_w, n, m, n_fp, m_fp, is_complex, training, adaptive_weights
    height_img = new_height_img
    width_img = new_width_img
    channel_img = new_channel_img
    filter_height = new_filter_height
    filter_width = new_filter_width
    num_filters = new_num_filters
    n_DnCNN_layers = new_n_DnCNN_layers
    n_DAMP_layers = new_n_DAMP_layers
    sampling_rate = new_sampling_rate
    BATCH_SIZE = new_BATCH_SIZE
    sigma_w = new_sigma_w
    n = new_n
    m = new_m
    n_fp = np.float32(n)
    m_fp = np.float32(m)
    is_complex=iscomplex#Just the default
    adaptive_weights=use_adaptive_weights
    training=new_training

def sensing_method(method_name,specifics):
    # a function which returns a sensing method with given parameters. a sensing method is a subclass of nn.Module
    return 1

#Form the measurement operators
def GenerateMeasurementOperators(mode):
    global is_complex
    if mode=='gaussian':
        is_complex=False
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m, n))# values that parameterize the measurement model. This could be the measurement matrix itself or the random mask with coded diffraction patterns.
        y_measured = tf.placeholder(tf.float32, [m, None])
        A_val_tf = tf.placeholder(tf.float32, [m, n])  #A placeholer is used so that the large matrix isn't put into the TF graph (2GB limit)

        def A_handle(A_vals_tf,x):
            return tf.matmul(A_vals_tf,x)

        def At_handle(A_vals_tf,z):
            return tf.matmul(A_vals_tf,z,adjoint_a=True)

    elif mode == 'complex-gaussian':
            is_complex = True
            A_val = np.complex64(1/np.sqrt(2.)*((1. / np.sqrt(m_fp) * np.random.randn(m,n))+1j*(1. / np.sqrt(m_fp) * np.random.randn(m,n))))
            y_measured = tf.placeholder(tf.complex64, [m, None])
            A_val_tf = tf.placeholder(tf.complex64, [m, n])  # A placeholer is used so that the large matrix isn't put into the TF graph (2GB limit)

            def A_handle(A_vals_tf, x):
                return tf.matmul(A_vals_tf, tf.complex(x,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)))

            def At_handle(A_vals_tf, z):
                return tf.matmul(A_vals_tf, z, adjoint_a=True)
    elif mode=='coded-diffraction':
        is_complex=True
        A_val = np.zeros([n, 1]) + 1j * np.zeros([n, 1])
        A_val[0:n] = np.exp(1j*2*np.pi*np.random.rand(n,1))#The random sign vector

        global sparse_sampling_matrix
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=list(zip(row_inds,rand_col_inds)) # 2.7 doesn't need the list(), 3.7.9 does to get list content
        vals=tf.ones(m, dtype=tf.complex64);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])

        A_val_tf = tf.placeholder(tf.complex64, [n, 1])
        def A_handle(A_val_tf, x):
            sign_vec = A_val_tf[0:n]
            signed_x = tf.multiply(sign_vec, tf.complex(x,tf.zeros([n,BATCH_SIZE],dtype=tf.float32)))
            signed_x = tf.reshape(signed_x, [height_img, width_img, BATCH_SIZE])
            signed_x=tf.transpose(signed_x)#Transpose because fft2d operates upon the last two axes
            F_signed_x = tf.fft2d(signed_x)
            F_signed_x=tf.transpose(F_signed_x)
            F_signed_x = tf.reshape(F_signed_x, [height_img * width_img, BATCH_SIZE])*1./np.sqrt(m_fp)#This is a different normalization than in Matlab because the FFT is implemented differently in Matlab
            out = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,F_signed_x,adjoint_a=False)
            return out

        def At_handle(A_val_tf, z):
            sign_vec=A_val_tf[0:n]
            z_padded = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,z,adjoint_a=True)
            z_padded = tf.reshape(z_padded, [height_img, width_img, BATCH_SIZE])
            z_padded=tf.transpose(z_padded)#Transpose because fft2d operates upon the last two axes
            Finv_z = tf.ifft2d(z_padded)
            Finv_z = tf.transpose(Finv_z)
            Finv_z = tf.reshape(Finv_z, [height_img*width_img, BATCH_SIZE])
            out = tf.multiply(tf.conj(sign_vec), Finv_z)*n_fp/np.sqrt(m)
            return out
    elif mode=='Fast-JL':#Measurement matrix close to a fast JL transform. True fast JL would use hadamard transform and a sparse sampling matrix with multiple nz elements per row
        is_complex=False
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)

        #global sparse_sampling_matrix
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=list(zip(row_inds,rand_col_inds)) # 2.7 doesn't need the list(), 3.7.9 does to get list content
        vals=tf.ones(m, dtype=tf.float32);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])

        A_val_tf = tf.placeholder(tf.float32, [n, 1])
        def A_handle(A_val_tf, x):
            sign_vec = A_val_tf[0:n]
            signed_x = tf.multiply(sign_vec, x)
            signed_x = tf.reshape(signed_x, [height_img*width_img, BATCH_SIZE])
            signed_x=tf.transpose(signed_x)#Transpose because dct operates upon the last axes
            F_signed_x = mydct(signed_x, type=2, norm='ortho')
            F_signed_x=tf.transpose(F_signed_x)
            F_signed_x = tf.reshape(F_signed_x, [height_img * width_img, BATCH_SIZE])*np.sqrt(n_fp/m_fp)
            out = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,F_signed_x,adjoint_a=False)
            return out

        def At_handle(A_val_tf, z):
            sign_vec=A_val_tf[0:n]
            z_padded = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,z,adjoint_a=True)
            z_padded = tf.reshape(z_padded, [height_img*width_img, BATCH_SIZE])
            z_padded=tf.transpose(z_padded)#Transpose because dct operates upon the last axes
            Finv_z = myidct(z_padded,type=2,norm='ortho')
            Finv_z = tf.transpose(Finv_z)
            Finv_z = tf.reshape(Finv_z, [height_img*width_img, BATCH_SIZE])
            out = tf.multiply(sign_vec, Finv_z)*np.sqrt(n_fp/m_fp)
            return out
    else:
        raise ValueError('Measurement mode not recognized')
    return [A_handle, At_handle, A_val, A_val_tf]

def GenerateMeasurementMatrix(mode):
    if mode == 'gaussian':
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m,n))  # values that parameterize the measurement model. This could be the measurement matrix itself or the random mask with coded diffraction patterns.
    elif mode=='complex-gaussian':
        A_val = np.complex64(1 / np.sqrt(2.) * ((1. / np.sqrt(m_fp) * np.random.randn(m, n)) + 1j * (1. / np.sqrt(m_fp) * np.random.randn(m, n))))
    elif mode=='coded-diffraction':
        A_val = np.zeros([n, 1]) + 1j * np.zeros([n, 1])
        A_val[0:n] = np.exp(1j*2*np.pi*np.random.rand(n,1))#The random sign vector
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=list(zip(row_inds,rand_col_inds)) # 2.7 doesn't need the list(), 3.7.9 does to get list content
        vals=tf.ones(m, dtype=tf.complex64);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    elif mode == 'Fast-JL':
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=list(zip(row_inds,rand_col_inds)) # 2.7 doesn't need the list(), 3.7.9 does to get list content
        vals=tf.ones(m, dtype=tf.float32);
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    else:
        raise ValueError('Measurement mode not recognized')
    return A_val

def mydct(x,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    y=tf.concat([x,tf.zeros([1,n],tf.float32)],axis=1)
    Y=tf.fft(tf.complex(y,tf.zeros([1,2*n],tf.float32)))
    Y=Y[:,:n]
    k = tf.complex(tf.range(n, dtype=tf.float32), tf.zeros(n, dtype=tf.float32))
    Y*=tf.exp(-1j*np.pi*k/(2.*n_fp))
    return tf.real(Y)/tf.sqrt(n_fp)*tf.sqrt(2.)
    # return tf.spectral.dct(x,type=2,norm='ortho')
def myidct(X,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE==1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    temp0=tf.reverse(X,[-1])
    temp1=tf.manip.roll(temp0,shift=1,axis=1)
    temp2=temp1[:,1:]
    temp3=tf.pad(temp2,[[0,0],[1,0]],"CONSTANT")
    Z=tf.complex(X,-temp3)
    k = tf.complex(tf.range(n,dtype=tf.float32),tf.zeros(n,dtype=tf.float32))
    Z*=tf.exp(1j*np.pi*k/(2.*n_fp))
    temp4=tf.real(tf.ifft(Z))
    even_new=temp4[:,0:n/2]
    odd_new=tf.reverse(temp4[:,n/2:],[-1])
    #https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices
    x=tf.reshape(
        tf.transpose(tf.concat([even_new, odd_new], axis=0)),
        [1,n])
    return tf.real(x)*tf.sqrt(n_fp)*1/tf.sqrt(2.)
def mydct_np(x,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    N=len(x)
    y=np.zeros(2*N)
    y[:N]=x
    Y=np.fft.fft(y)[:N]
    k=np.float32(range(N))
    Y*=np.exp(-1j*np.pi*k/(2*N))/np.sqrt(N/2.)
    return Y.real
def myidct_np(X,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    #https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/
    N=len(X)
    Z=X-1j*np.append([0.],np.flip(X,0)[:N-1])
    k = np.float32(range(N))
    Z*=np.exp(1j*np.pi*k/(2*N))
    temp=np.real(np.fft.ifft(Z))
    x=np.zeros(X.size)
    even_new= temp[0:N/2]
    odd_new=np.flip(temp[N/2:],0)
    x[0::2] =even_new
    x[1::2]=odd_new
    return np.real(x)*np.sqrt(N/2.)