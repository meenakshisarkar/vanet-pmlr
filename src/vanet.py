import os
import tensorflow as tf
import keras
import numpy as np
from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class VANET(object):
    def __init__(self, image_size=128, batch_size= 32, c_dim=3, timesteps=10, predict=10,
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
        self.xt_shape= [batch_size, timesteps, image_size[0], image_size[1], c_dim]
        self.predict_shape=[batch_size, predict, image_size[0], image_size[1], c_dim]
        
        self.create_model()
    
    
    def create_model(self):
        
