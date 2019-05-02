import os
import tensorflow as tf
import keras
import numpy as np
from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class VCPROM(object):
    def __init__(self, image_size=128, batch_size= 32, c_dim=3, timesteps=10, predict=10, checkpoint_dir=None, training=True):
        self.image_size= image_size
        self.batch_size= batch_size
        self.c_dim= c_dim
        self.timesteps= timesteps
        self.predict= predict
        self.checkpoint_dir=
