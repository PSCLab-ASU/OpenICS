# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

import os
import shutil
import tensorflow as tf
import scipy.misc
import numpy as np


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img


def print_hparams(hparams):
    print ('')
    for temp in dir(hparams):
        if temp[:1] != '_':
            print('{0} = {1}'.format(temp, getattr(hparams, temp)))
    print ('')


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)


def get_ckpt_path(ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = os.path.join(ckpt_dir,
                                 ckpt.model_checkpoint_path)
    else:
        ckpt_path = None
    return ckpt_path


def try_restore(hparams, sess, model_saver):
    """Attempt to restore variables from checkpoint"""
    ckpt_path = get_ckpt_path(hparams.ckpt_dir)
    if ckpt_path:  # if a previous ckpt exists
        model_saver.restore(sess, ckpt_path)
        start_epoch = int(ckpt_path.split('/')[-1].split('-')[-1])
        print('Succesfully loaded model from {0} at counter = {1}'.format(
            ckpt_path, start_epoch))
    else:
        print('No checkpoint found')
        start_epoch = -1
    return start_epoch



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0 / (fan_in + fan_out))
    high = constant*np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
