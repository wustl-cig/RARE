from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow.keras.backend as k
import sys
import os
import shutil
import time
import math
import numpy as np
import logging
import tensorflow as tf
from util import *

from abc import ABC, abstractmethod
from collections import OrderedDict
from Regularizers.nets import *

############## Basis Class ##############

class RegularizerClass(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def deartifacts_A2A(self, s, pin, useNoise, clip):
        pass

    @abstractmethod
    def eval(self,z,step,pin):
        pass

############## Regularizer Class ##############
class DnCNN3DClass(RegularizerClass):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    :param kwargs: args passed to create_net function. 
    """
   
    def __init__(self, sigSize, model_path, num_layers=10, img_channels=2, truth_channels=2, dilation_rate=None, cnnmode='NOBN'):
        tf.reset_default_graph()

        # basic variables
        self.img_channels = img_channels
        self.truth_channels = truth_channels

        # reused variables
        self.nx = sigSize[0]
        self.ny = sigSize[1]
        self.np = sigSize[2]


        self.model = dncnn(input_shape=(self.np, self.nx, self.ny, self.truth_channels), output_channel=self.truth_channels)
        self.model.summary()
        # placeholders for input x
        self.x = tf.placeholder("float", shape=[None, self.np, self.nx, self.ny, self.truth_channels]) 
        # variables need to be calculated
        self.recons = self.model(self.x)
        self.vars = self._get_vars()
        self.convolutional_operators = [v for v in self.vars if 'kernel:' in v.name]

        # load pretrained net to sess
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        init = tf.global_variables_initializer()
        k.set_session(self.sess)
        self.sess.run(init)
        self.model.load_weights(model_path)
    
    def _get_vars(self):
        lst_vars = []
        for v in tf.global_variables():
            lst_vars.append(v)
        return lst_vars

    def init(self):
        p = np.zeros([self.nx, self.ny, self.np], dtype=np.complex128)
        return p

    def deartifacts_A2A(self, s, pin=None, useNoise=False, clip=False):
        s_abs = np.abs(s)
        s_phase = np.angle(s)
        s_norm, s_abs_min, s_abs_max  = to_double_all(s_abs)
        s_norm = s_norm * np.exp(1j*s_phase)
        stemp = s_norm.transpose([2,0,1])
        if len(s.shape) == 3:
            # reshape
            num_real,num_imag = [1,1]
            stemp = np.expand_dims(np.expand_dims(stemp,axis=0),axis=4)
            stemp_multi = np.concatenate((stemp.real*num_real,stemp.imag*num_imag),axis=4)       
            xtemp = self.sess.run(self.recons, feed_dict={self.x: stemp_multi, 
                                                            k.learning_phase(): 0})                                                        
        else:
            print('Incorrect s.shape')
            exit()
        xtemp = xtemp.squeeze().transpose([1,2,0,3])
        xtemp = (xtemp[...,0]/num_real) + 1j*(xtemp[...,1]/num_imag)

        if useNoise:
            noise = xtemp
        else:
            xtemp_phase = np.angle(xtemp)
            xtemp = (np.abs(xtemp) * s_abs_max + s_abs_min) * np.exp(1j*xtemp_phase)        
            noise = (s - xtemp)
        return noise

    def eval(self, x):
        return 0

    def name(self):
        return 'A2A'

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver(var_list=self.vars)
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)