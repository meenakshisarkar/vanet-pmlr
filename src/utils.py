"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
# from skimage.color import rgb2gray
import scipy.misc
import numpy as np
import os
import glob
from PIL import Image
np.random.seed(77)

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


# def save_images(images, size, image_path):
#   return imsave((images)*255., size, image_path)
def save_images(images, size, image_path):
    return imsave(np.int32((images+1.)*255./2.), size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  return imageio.imwrite(path, merge(images, size))


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 




def load_kth_data(vid_dir, dir_length, resize_shape, K, T):
    vid_frames = []
    low = int(dir_length[0])
    high = int(dir_length[1]) - K - T + 1
    assert low <= high, 'video length shorter than K+T_{}'.format(vid_dir)
    stidx = np.random.randint(low, high)
    for t in range(0, K+T):  
        fname =  "{}/img_{:06d}.png".format(vid_dir.split('#')[0], t+stidx)
        im = imageio.imread(fname)
        # im=rgb2gray(im)
        im=Image.fromarray(im).resize((resize_shape[1], resize_shape[0]))
        im= im.convert('L')
        im= np.expand_dims(im, axis=0)
        vid_frames.append(im/255.)
    vid = np.concatenate(vid_frames, axis=0)
    vid= np.expand_dims(vid, axis=-1)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    return vid, diff, accel

def load_kitti_data(vid_dir, length, resize_shape, K, T):
    vid_frames = []
    low=length[0]
    high=length[1]-(K+T+5)+1

    assert low <= high, 'video length shorter than K+T'
    stidx = np.random.randint(low, high)
    for t in range(0, K+T):  
        fname =  "{}/{:010d}.png".format(vid_dir.split('#')[0], t+stidx)
        im = imageio.imread(fname)
        im=Image.fromarray(im).resize((resize_shape[1], resize_shape[0]))
        im= np.expand_dims(im, axis=0)
        # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
        vid_frames.append(im/255.)
    vid = np.concatenate(vid_frames, axis=0)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    return vid, diff, accel

def load_bair_data(vid_dir, K, T):
    vid_frames = []
    seq_len = K+T
    for i in range(seq_len):
        fname = "{}/{}.png".format(vid_dir, i)
        # im = scipy.misc.imread(fname).reshape(1, 64, 64, 3)
        im = imageio.imread(fname).reshape(1, 64, 64, 3)
        vid_frames.append(im/255.)
    vid = np.concatenate(vid_frames, axis=0)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    return vid, diff, accel

def load_bair_towel_data(vid_dir, K, T):
    vid_frames = []
    seq_len = K+T
    for i in range(seq_len):
        fname = "{}/{}.png".format(vid_dir, i)
        # im = scipy.misc.imread(fname).reshape(1, 64, 64, 3)
        im = imageio.imread(fname)
        im=Image.fromarray(im).resize((64, 64))
        im= np.expand_dims(im, axis=0)
        vid_frames.append(im/255.)
    vid = np.concatenate(vid_frames, axis=0)
    diff = vid[1:K, ...] - vid[:K-1, ...]
    accel = diff[1:, ...] - diff[:-1, ...]
    return vid, diff, accel

