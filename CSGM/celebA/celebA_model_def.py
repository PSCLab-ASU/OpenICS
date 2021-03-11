# pylint: disable = C0103, C0111, C0301, R0914

"""Model definitions for celebA

This file is partially based on
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/main.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

import os
import sys
import tensorflow.compat.v1 as tf

#sys.path.append(os.path.join(os.path.dirname(__file__), '..')) #file has benn moved
#from celebA_dcgan import model_def as celebA_dcgan_model_def #was moved
import CSGM.dcgan.model_def as celebA_dcgan_model_def 

def dcgan_discrim(x_hat_batch, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    x_hat_image = tf.reshape(x_hat_batch, [-1, hparams.input_size, hparams.input_size, hparams.input_channels])
    all_zeros = tf.zeros([64, hparams.input_size, hparams.input_size, hparams.input_channels])
    discrim_input = all_zeros + x_hat_image

    model_hparams = celebA_dcgan_model_def.Hparams(hparams)
    prob, _ = celebA_dcgan_model_def.discriminator(model_hparams, discrim_input, train=False, reuse=False)
    prob = tf.reshape(prob, [-1])
    prob = prob[:hparams.batch_size]

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return prob, restore_dict, restore_path



def dcgan_gen(z, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    z_full = tf.zeros([64, 100]) + z

    model_hparams = celebA_dcgan_model_def.Hparams(hparams)

    x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
    x_hat_batch = tf.reshape(x_hat_full[:hparams.batch_size], [hparams.batch_size, hparams.input_size*hparams.input_size*hparams.input_channels])

    restore_vars = celebA_dcgan_model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)


    return x_hat_batch, restore_dict, restore_path
