import sys
import math
import numpy as np 
import tensorflow as tf
from joblib import Parallel, delayed

from tensorflow.python.framework import ops

from utils import *




def batch_norm(inputs, name, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
                                      reuse=reuse,scope=name,scale=True)

def cross_conv(input,kernal, reuse=False, name= None, padding= 'SAME'):
  
  _input= input[tf.newaxis,:,:,:,:]
  _input= tf.reshape(tf.transpose(_input, [0,2,3,4,1]),[1,_input.shape[2], _input.shape[3],-1])
  _input= _input[:,:,:,:,tf.newaxis]

  kernal= tf.reshape(tf.transpose(kernal, [1,2,3,0]),[kernal.shape[1], kernal.shape[2],-1])
  kernal= kernal[:,:,:,tf.newaxis,tf.newaxis]
  # print kernal.shape, _input.shape
  with tf.variable_scope(name, reuse=reuse):
        output= tf.nn.convolution(_input, kernal, padding= padding, data_format= "NDHWC")
        output= tf.squeeze(output)
        output= tf.reshape(output,[input.shape[1],input.shape[2],input.shape[3],input.shape[0]])
        output= tf.transpose(output,[3,0,1,2])
        # print output.shape
  return output



def convOp(input, kernal, reuse=False,name=None):
  return cross_conv(input, kernal, reuse, name)

def convOp_mod(input, kernal, reuse=False,name=None):
  with tf.variable_scope(name):
    return input+kernal

def convOp_mod1(input, kernal, reuse=False,name=None):
  with tf.variable_scope(name):
    norm_kernal= tf.nn.l2_normalize(kernal, axis=[3],epsilon=1e-12)
    return input+tf.multiply(input, norm_kernal)

  

    



def conv2d(input_, output_dim, 
            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
            name="conv2d", reuse=False, padding='SAME'):
   with tf.variable_scope(name, reuse=reuse):
     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                         initializer=tf.contrib.layers.xavier_initializer())
     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
 
     biases = tf.get_variable('biases', [output_dim],
                              initializer=tf.constant_initializer(0.0))
     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
 
     return conv


def deconv2d(input_, output_shape,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="deconv2d", reuse=False, with_w=False, padding='SAME'):
  with tf.variable_scope(name, reuse=reuse):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_h, output_shape[-1],
                              input_.get_shape()[-1]],
                        initializer=tf.contrib.layers.xavier_initializer())
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w,
                                      output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1],
                                      padding=padding)

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                          strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def relu(x):
  return tf.nn.relu(x)


def tanh(x):
  return tf.nn.tanh(x)


def shape2d(a):
  """
  a: a int or tuple/list of length 2
  """
  if type(a) == int:
      return [a, a]
  if isinstance(a, (list, tuple)):
      assert len(a) == 2
      return list(a)
  raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


def UnPooling2x2ZeroFilled(x):
  out = tf.concat(axis=3, values=[x, tf.zeros_like(x)])
  out = tf.concat(axis=2, values=[out, tf.zeros_like(out)])

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    return tf.reshape(out, out_size)
  else:
    sh = tf.shape(x)
    return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


def MaxPooling(x, shape, stride=None, padding='VALID'):
  """
  MaxPooling on images.
  :param input: NHWC tensor.
  :param shape: int or [h, w]
  :param stride: int or [h, w]. default to be shape.
  :param padding: 'valid' or 'same'. default to 'valid'
  :returns: NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


#@layer_register()
def FixedUnPooling(x, shape):
  """
  Unpool the input with a fixed mat to perform kronecker product with.
  :param input: NHWC tensor
  :param shape: int or [h, w]
  :returns: NHWC tensor
  """
  shape = shape2d(shape)
  
  # a faster implementation for this special case
  return UnPooling2x2ZeroFilled(x)


def stgdl(gen_frames, gt_frames, alpha, image_size, channel_no):
  """
  Calculates the sum of GDL losses between the predicted and gt frames.
  @param gen_frames: The predicted frames at each scale.
  @param gt_frames: The ground truth frames at each scale
  @param alpha: The power to which each gradient term is raised.
  @param image_size: tuple for image shape (h,w)
  @return: The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(channel_no), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  # [[[-1]],[[1]]]
  filter_t = tf.stack([tf.expand_dims(tf.expand_dims(neg,0),0), tf.expand_dims(tf.expand_dims(pos,0),0)])
  print filter_t.shape, filter_x.shape, filter_y.shape
  strides1 = [1, 1, 1, 1]  # stride of (1, 1)
  strides2= [1, 1, 1, 1, 1] #stride of (1,1,1) for conv3D
  padding = 'SAME'

  gen_dt = tf.abs(tf.nn.conv3d(gen_frames, filter_t,strides2, padding=padding))
  gt_dt = tf.abs(tf.nn.conv3d(gt_frames, filter_t, strides2, padding=padding))
  # gen_frames= tf.reshape(gen_frames, [-1, image_size,image_size,channel_no])
  # gt_frames= tf.reshape(gt_frames, [-1, image_size,image_size,channel_no])

  gen_dx = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(tf.reshape(gen_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_x, strides1, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(tf.reshape(gt_frames, [-1, image_size[0],image_size[1],channel_no]),
                                                             filter_y, strides1, padding=padding))
  
  

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)
  grad_diff_t = tf.abs(gt_dt-gen_dt)
  grad_diff_t= tf.reshape(grad_diff_t,[-1, image_size, image_size,channel_no])

  spatial_loss = (tf.reduce_mean((grad_diff_t**alpha + grad_diff_x ** alpha + grad_diff_y ** alpha)))

  gen_ddx = tf.abs(tf.nn.conv2d(gen_dx, filter_x, strides1, padding=padding))
  gen_ddy = tf.abs(tf.nn.conv2d(gen_dy, filter_y, strides1, padding=padding))
  gen_ddt = tf.abs(tf.nn.conv3d(gen_dt, filter_t, strides2, padding=padding))
  gt_ddx = tf.abs(tf.nn.conv2d(gt_dx, filter_x, strides1, padding=padding))
  gt_ddy = tf.abs(tf.nn.conv2d(gt_dy, filter_y, strides1, padding=padding))
  gt_ddt = tf.abs(tf.nn.conv3d(gt_dt, filter_t, strides2, padding=padding))

  grad_diff_dx = tf.abs(gt_ddx - gen_ddx)
  grad_diff_dy = tf.abs(gt_ddy - gen_ddy)
  grad_diff_dt = tf.abs(gt_ddt - gen_ddt)
  grad_diff_dt = tf.reshape(grad_diff_dt, [-1, image_size, image_size, channel_no])

  velocity_loss = (tf.reduce_mean((grad_diff_dt**alpha + grad_diff_dx ** alpha + grad_diff_dy ** alpha)))

  stgdl_loss= spatial_loss+velocity_loss           #, [gen_frames.shape[0], gen_frames.shape[1], image_size, image_size,channel_no])

  # condense into one tensor and avg
  return stgdl_loss


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0,
           reuse=False, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

