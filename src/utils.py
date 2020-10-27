"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np
import os
import glob

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def save_images(images, size, image_path):
  return imsave((images)*255., size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))


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


def load_kth_data(f_name, data_path, image_size, K, T): 
  flip = np.random.binomial(1,.5,1)[0]
  tokens = f_name.split()
  vid_path = data_path + tokens[0] + "_uncomp.avi"
  vid = imageio.get_reader(vid_path,"ffmpeg")
  low = int(tokens[1])
  high = np.min([int(tokens[2]),vid.get_length()])-K-T+1
  if low == high:
    stidx = 0 
  else:
    if low >= high: print(vid_path)
    stidx = np.random.randint(low=low, high=high)
  seq = np.zeros((K+T, image_size, image_size, 1), dtype="float32")
  # print seq.shape
  for t in xrange(0,K+T):
    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx+t),
                       (image_size,image_size)),
                       cv2.COLOR_RGB2GRAY)
    seq[t,:,:] = inverse_transform(transform(img[:,:,None]))

  # if flip == 1:
  #   seq = seq[:-1,:,:]

  diff = np.zeros((K-1, image_size, image_size, 1), dtype="float32")
  for t in xrange(1,K):
    prev = seq[t-1,:,:]
    next = seq[t,:,:]
    diff[t-1,:,:] = next.astype("float32")-prev.astype("float32")
  accel= np.zeros((K-2, image_size,image_size,1),dtype="float32")
  for t in xrange(1,K-1):
    prev_diff= diff[t-1,:, :]
    next_diff= diff[t,:,:]
    accel[t-1,:,:]= next_diff.astype("float32")-prev_diff.astype("float32")

  return seq, diff, accel


def load_s1m_data(f_name, data_path, trainlist, K, T):
  flip = np.random.binomial(1,.5,1)[0]
  vid_path = data_path + f_name
  img_size = [240,320]

  while True:
    try:
      vid = imageio.get_reader(vid_path,"ffmpeg")
      low = 1
      high = vid.get_length()-K-T+1
      if low == high:
        stidx = 0
      else:
        stidx = np.random.randint(low=low, high=high)
      seq = np.zeros((img_size[0], img_size[1], K+T, 3),
                     dtype="float32")
      for t in xrange(K+T):
        img = cv2.resize(vid.get_data(stidx+t),
                         (img_size[1],img_size[0]))[:,:,::-1]
        seq[:,:,t] = transform(img)

      if flip == 1:
        seq = seq[:,::-1]

      diff = np.zeros((img_size[0], img_size[1], K-1, 1),
                      dtype="float32")
      for t in xrange(1,K):
        prev = inverse_transform(seq[:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq[:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff[:,:,t-1,0] = (next.astype("float32")-prev.astype("float32"))/255.
      break
    except Exception:
      # In case the current video is bad load a random one 
      rep_idx = np.random.randint(low=0, high=len(trainlist))
      f_name = trainlist[rep_idx]
      vid_path = data_path + f_name
  return seq, diff

def load_kitti_data(vid_dir, data_path, resize_h, K, T):
  
  """
  Arguments:

    vid_dir: date_drive folder e.g. 2011_09_26_drive_00002_sync
    data_path: base path
    resize_h: height to which each frame would be resized, resize_w is computed based on aspect ratio
    K: num input time steps
    T: num output time steps

  Returns:

      seq: K+T length video sequence
      diff: velocity map
      accel: acceleration map
  """
  
  vid_path = os.path.join(data_path, vid_dir)
  img_files = glob.glob(os.path.join(vid_path, 'image_00/data/*.png'))
  imgs = [imageio.imread(img_file)[np.newaxis, :, :] for img_file in sorted(img_files)]
  vid = np.concatenate(imgs, axis=0)
  
  low = 0
  high = vid.shape[0] - K - T + 1
  assert low <= high, 'video length shorter than K+T'
  
  stidx = np.random.randint(low, high)
  img_h, img_w = vid.shape[1:]
  r = resize_h / img_h
  resize_shape = (resize_h, int(img_w * r))
  
  seq = np.zeros((K+T, *resize_shape, 1), dtype="float32")
  
  for t in range(0, K+T):
    img = cv2.resize(vid[stidx + t, :, :], resize_shape[::-1])
    seq[t, :, :] = img[:, :, np.newaxis]
  
  diff = np.zeros((K-1, *resize_shape, 1), dtype="float32")
  
  for t in range(1, K):
    prev = seq[t-1, :, :]
    next = seq[t, :, :]
    diff[t-1, :, :] = next.astype('float32') - prev.astype('float32')
  
  accel= np.zeros((K-2, *resize_shape,1), dtype="float32")
  
  for t in range(1,K-1):
    prev_diff= diff[t-1,:, :]
    next_diff= diff[t,:,:]
    accel[t-1,:,:]= next_diff - prev_diff  
  
  return seq, diff, accel
