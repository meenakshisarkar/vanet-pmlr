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
        vel_LSTM = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                 [3, 3], self.filters*4)
        predict = forward_model(self.velocity, self.accelaration, self.xt, cell)
        self.G = tf.concat(axis=3, values=predict)

    def forward_model(self, diff_in, xt, cell):
        state_vel = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 512])
        reuse_vel = False
        # Encoder
        for t in xrange(self.timesteps-1):
            h_vel_out, vel_state, vel_res_in = self.vel_enc(diff_in[:, :, :, t, :], reuse=reuse_vel)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse_vel)
            reuse_vel = True
        for t in xrange(self.timesteps-2):
            enc_h, res_m = self.motion_enc(diff_in[:, :, :, t, :], reuse=reuse_vel)
            h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse_vel)
            reuse_vel = True

        pred = []

    def vel_enc(self, vel_in, vel_state, vel_LSTM , reuse):
        vel_res_in = []
        conv1 = relu(conv2d(vel_in, output_dim=self.filters, k_h=5, k_w=5,
                        d_h=1, d_w=1, name='dyn_conv1', reuse=reuse))
        vel_res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2,2])

        conv2 = relu(conv2d(pool1, output_dim=self.filters*2, k_h=5, k_w=5,
                        d_h=1, d_w=1, name='dyn_conv2',reuse=reuse))
        vel_res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2,2])

        conv3 = relu(conv2d(pool2, output_dim=self.filters*4, k_h=7, k_w=7,
                        d_h=1, d_w=1, name='dyn_conv3',reuse=reuse))
        vel_res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2,2])
        h1_state, state = vel_LSTM(conv3, state, scope= 'vel_lstm1', reuse=reuse_vel)
        h2_state, state = vel_LSTM(h1_state, state, scope= 'vel_lstm2', reuse=reuse_vel)
        h3_state, state = vel_LSTM(h2_state, state, scope= 'vel_lstm3', reuse=reuse_vel)
        return h_vel_out, vel_state, vel_res_in
    
    def acc_end(self):
    

    def content_enc(self):
    

    def dec(self):
