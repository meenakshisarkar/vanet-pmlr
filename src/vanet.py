import os
import tensorflow as tf
import keras
import numpy as np
from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class VANET(object):
    def __init__(self, image_size=[128, 128], batch_size=32, c_dim=3, timesteps=10, predict=10,
                 checkpoint_dir=None, training=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.timesteps = timesteps
        self.predict = predict
        self.checkpoint_dir = checkpoint_dir
        self.training = training

        self.vel_shape = [batch_size, timesteps - 1, image_size[0], image_size[1], c_dim]
        self.acc_shape = [batch_size, timesteps - 2, image_size[0], image_size[1], c_dim]
        self.xt_shape = [batch_size, image_size[0], image_size[1], c_dim]
        self.gt_shape = [batch_size, predict, image_size[0], image_size[1], c_dim]

        self.create_model()

    def create_model(self):
        self.velocity = tf.placeholder(tf.float32, self.vel_shape, name='velocity')
        self.accelaration = tf.placeholder(tf.float32, self.acc_shape, name='accelaration')
        self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.predict = tf.placeholder(tf.float32, self.predict_shape, name='predict')
        self.vel_LSTM = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                     [3, 3], self.filters * 4)
        self.acc_LSTM = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                          [3, 3], self.filters * 4)
        predict = forward_model(self.velocity, self.accelaration, self.xt, cell)
        self.G = tf.concat(axis=3, values=predict)

    def forward_model(self, vel_in, acc_in, xt, vel_LSTM, acc_LSTM):
        vel_state = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 512])
        acc_state = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 512])
        reuse_vel = False
        reuse_acc= False

        # Encoder
        for t in xrange(self.timesteps - 1):
            h_vel_out, vel_state, vel_res_in = self.vel_enc(vel_in[:, :, :, t, :], vel_state, vel_LSTM, reuse=reuse_vel)
            if t<=self.timesteps-2:
                h_acc_out, acc_state, acc_res_in = self.acc_enc(acc_in[:, :, :, t, :], acc_state, acc_LSTM, reuse=reuse_acc)
            reuse_vel = True
            reuse_acc = True
        
        pred = []
        for t in xrange(self.predict):
            if t==0:
                h_con_state, con_res_in= self.content_enc(xt, reuse=False)
                



    def vel_enc(self, vel_in, vel_state, vel_LSTM, reuse):
        vel_res_in = []
        conv1 = relu(conv2d(vel_in, output_dim=self.filters, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='vel_conv1', reuse=reuse))
        vel_res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='vel_conv2', reuse=reuse))
        vel_res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=7, k_w=7,
                            d_h=1, d_w=1, name='vel_conv3', reuse=reuse))
        vel_res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        h1_state, vel_state = vel_LSTM(pool3, vel_state, scope='vel_lstm1', reuse=reuse)
        h2_state, vel_state = vel_LSTM(h1_state, vel_state, scope='vel_lstm2', reuse=reuse)
        h_vel_out, vel_state = vel_LSTM(h2_state, vel_state, scope='vel_lstm3', reuse=reuse)
        return h_vel_out, vel_state, vel_res_in

    def acc_end(self, acc_in, acc_state, acc_LSTM, reuse):
        acc_res_in = []
        conv1 = relu(conv2d(acc_in, output_dim=self.filters, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='acc_conv1', reuse=reuse))
        acc_res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='acc_conv2', reuse=reuse))
        acc_res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=7, k_w=7,
                            d_h=1, d_w=1, name='acc_conv3', reuse=reuse))
        acc_res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        h1_state, acc_state = acc_LSTM(pool3, acc_state, scope='acc_lstm1', reuse=reuse)
        h2_state, acc_state = acc_LSTM(h1_state, acc_state, scope='acc_lstm2', reuse=reuse)
        h_acc_out, acc_state = acc_LSTM(h2_state, acc_state, scope='acc_lstm3', reuse=reuse)
        return h_acc_out, acc_state, acc_res_in

    def content_enc(self,xt,reuse):
        con_res_in = []
        conv1 = relu(conv2d(xt, output_dim=self.filters, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='con_conv1', reuse=reuse))
        con_res_in.append(conv1)
        pool1 = MaxPooling(conv1, [2, 2])

        conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                            d_h=1, d_w=1, name='con_conv2', reuse=reuse))
        con_res_in.append(conv2)
        pool2 = MaxPooling(conv2, [2, 2])

        conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=7, k_w=7,
                            d_h=1, d_w=1, name='con_conv3', reuse=reuse))
        con_res_in.append(conv3)
        pool3 = MaxPooling(conv3, [2, 2])
        return pool3, con_res_in

    def dec(self):
        return
    def conv_layer(self):
        return
