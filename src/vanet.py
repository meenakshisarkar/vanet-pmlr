import os
import tensorflow as tf
import keras
import numpy as np
from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class VANET(object):
    def __init__(self, image_size=[128,128], batch_size= 32, c_dim=3, timesteps=10, predict=10,
                 checkpoint_dir=None, training=True):
        self.image_size= image_size
        self.batch_size= batch_size
        self.c_dim= c_dim
        self.timesteps= timesteps
        self.predict= predict
        self.checkpoint_dir= checkpoint_dir
        self.training= training
        
        self.vel_shape= [batch_size, timesteps-1, image_size[0], image_size[1], c_dim]
        self.acc_shape= [batch_size, timesteps-2, image_size[0], image_size[1], c_dim]
        self.xt_shape= [batch_size, image_size[0], image_size[1], c_dim]
        self.gt_shape=[batch_size, predict, image_size[0], image_size[1], c_dim]
        
        self.create_model()
    
    
        def create_model(self):
        self.velocity = tf.placeholder(tf.float32, self.vel_shape, name='velocity')
        self.accelaration = tf.placeholder(tf.float32, self.acc_shape, name='accelaration')
        self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.predict = tf.placeholder(tf.float32, self.predict_shape, name='predict')
        cell = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], 256)
        predict = forward_model(self.velocity, self.accelaration, self.xt, cell)
        self.G = tf.concat(axis=3, values=predict)

    def forward_model(self, diff_in, xt, cell):
        state_vel = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 512])
        reuse_vel = False
        # Encoder
        for t in xrange(self.timesteps-1):
            enc_h, res_m = self.motion_enc(diff_in[:, :, :, t, :], reuse=reuse_vel)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse_vel)
            reuse_vel = True
        for t in xrange(self.timesteps-2):
            enc_h, res_m = self.motion_enc(diff_in[:, :, :, t, :], reuse=reuse_vel)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse_vel)
            reuse_vel = True

        pred = []

    def vel_enc(self):
        cell1 = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], 256)
        cell2 = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], 256)

        

    def acc_end(self):
    

    def content_enc(self):
    

    def dec(self):


        
