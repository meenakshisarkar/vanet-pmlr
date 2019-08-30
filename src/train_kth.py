import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from vanet import VANET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu):
  data_path = "../data/KTH/"
  f = open(data_path+"train_data_list_trimmed_edited.txt","r")
  trainfiles = f.readlines()
  margin = 0.3 
  updateD = True
  updateG = True
  iters = 0
  prefix  = ("KTH_VANET"
          + "_image_size="+str(image_size)
          + "_K="+str(K)
          + "_T="+str(T)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../models/"+prefix+"/"
  samples_dir = "../samples/"+prefix+"/"
  summary_dir = "../logs/"+prefix+"/"

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  if not exists(samples_dir):
    makedirs(samples_dir)
  if not exists(summary_dir):
    makedirs(summary_dir)

  if gpu==0:
    gpus= False           #checking for GPU availability
  else: 
    gpus= True


  with tf.device("/cpu:0"):             #Selecting cpu or gpu "/gpu:%d"%gpu[0] if gpus else 
    model = VANET(image_size=[image_size,image_size], c_dim=1,
                  timesteps=K, batch_size=batch_size, F=T, checkpoint_dir=checkpoint_dir)
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars)
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha*model.reconst_loss+beta*model.L_gen, var_list=model.g_vars)


  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options if gpus else None)) as sess:                             #(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options if gpus else None))

    tf.global_variables_initializer().run()

    success_load_model=model.load(sess, checkpoint_dir)
    print success_load_model[0]

    if success_load_model[0]:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")



    g_sum = tf.summary.merge([model.L_p_sum,
                              model.L_stgdl_sum, model.L_sum,
                              model.L_Gen_sum])
    d_sum = tf.summary.merge([model.d_loss_real_sum,
                              model.d_loss_fake_sum, model.d_loss_sum])
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    counter = iters+1
    start_time = time.time()

    with Parallel(n_jobs=batch_size) as parallel:
      while iters < num_iter:
        mini_batches = get_minibatches_idx(len(trainfiles), batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch  = np.zeros((batch_size, K+T, image_size, image_size,
                                    1), dtype="float32")
            diff_batch = np.zeros((batch_size,K-1, image_size, image_size,
                                    1), dtype="float32")
            accel_batch = np.zeros((batch_size,K-2, image_size, image_size,
                                    1), dtype="float32")
            t0 = time.time()
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            paths = np.repeat(data_path, batch_size,axis=0)
            tfiles = np.array(trainfiles)[batchidx]
            shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
            output = parallel(delayed(load_kth_data)(f, p,img_sze, k, t)
                                                 for f,p,img_sze,k,t in zip(tfiles,
                                                                            paths,
                                                                            shapes,
                                                                            Ks, Ts))
            print seq_batch[1].shape, output[1][0].shape
            for i in xrange(batch_size):
              seq_batch[i] = output[i][0]
              diff_batch[i] = output[i][1]
              accel_batch[i] = output[i][2]
            print "I am at checkpoint batcave"
###################### need to change the input to the model and the indexing of the input images needs to be correct.model.target: seq_batch
            
            if updateD:
              print "here there"
              _, summary_str = sess.run([d_optim, d_sum],
                                         feed_dict={model.velocity: diff_batch,
                                                    model.accelaration: accel_batch,
                                                    model.xt: seq_batch[:,K-1,:,:,:],
                                                    model.target: seq_batch})
              print "ola"
              writer.add_summary(summary_str, counter)
            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                         feed_dict={model.velocity: diff_batch,
                                                    model.accelaration: accel_batch,
                                                    model.xt: seq_batch[:,K-1,:,:],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)
            print "I am at checkpoint gotham"

            

            errD_fake = model.d_loss_fake.eval({model.velocity: diff_batch,
                                                  model.accelaration: accel_batch,
                                                  model.xt: seq_batch[:,K-1,:,:,:],
                                                  model.target: seq_batch})
            errD_real = model.d_loss_real.eval({model.velocity: diff_batch,
                                                  model.accelaration: accel_batch,
                                                  model.xt: seq_batch[:,K-1,:,:,:],
                                                  model.target: seq_batch})
            errG = model.L_gen.eval({model.velocity: diff_batch,
                                          model.accelaration: accel_batch,
                                          model.xt: seq_batch[:,K-1,:,:,:],
                                          model.target: seq_batch})

            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1.-margin) or errD_real > (1.-margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True

            counter += 1
  
            print(
                "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f" 
                % (iters, time.time() - start_time, errD_fake+errD_real,errG)
            )

            if np.mod(counter, 100) == 1:
              samples = sess.run([model.G],
                                  feed_dict={model.velocity: diff_batch,
                                          model.accelaration: accel_batch,
                                          model.xt: seq_batch[K-1,:,:,:],
                                          model.target: seq_batch})[0]
              samples = samples[0]
              sbatch  = seq_batch[0,K:,:,:]
              samples = np.concatenate((samples,sbatch), axis=0)
              print("Saving sample ...")
              save_images(samples[:,:,:,::-1], [2, T], 
                          samples_dir+"train_%s.png" % (iters))
            if np.mod(counter, 500) == 2:
              model.save(sess, checkpoint_dir, counter)
  
            iters += 1

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=2, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.02, help="GAN loss weight")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=128, help="Mini-batch size")
  parser.add_argument("--K", type=int, dest="K",
                      default=3, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=1, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=False,
                      default=0, help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))