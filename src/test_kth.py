import os
import cv2
import sys
import time
import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from vanet_v2 import VANET_v2 
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(prefix, image_size, K, T, gpu):
  data_path = "../data/KTH/"
  f = open(data_path+"test_data_list_walk.txt","r")
  testfiles = f.readlines()
  c_dim = 1
  iters = 100
  samples_dir= "../results/images/test18_14_walking"
  prefix == "Paper_model"
  checkpoint_dir="../models/KTH/Paper_model"
  best_model = "VANET.model-100002"
#   else:
#     checkpoint_dir = "../models/"+prefix+"/"
#     best_model = None # will pick last model

  with tf.device("/cpu:0"): #"/gpu:%d"%gpu[0]):
    model = VANET_v2(image_size=[image_size, image_size], c_dim=1,
                         timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir, training=False)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=None)) as sess:  #add gpu_option

    tf.global_variables_initializer().run()

    loaded, model_name = model.load(sess, checkpoint_dir,best_model)

    if loaded:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed... exitting")
      return

    quant_dir = "../results/quantitative/KTH/"+prefix+"/"
    save_path = quant_dir+"results_model="+model_name+".npz"
    if not exists(quant_dir):
      makedirs(quant_dir)


    vid_names = []
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))
    for i in xrange(len(testfiles)):
      tokens = testfiles[i].split()
      vid_path = data_path+tokens[0]+"_uncomp.avi"
      while True:
        try:
          vid = imageio.get_reader(vid_path,"ffmpeg")
          break
        except Exception:
          print("imageio failed loading frames, retrying")

    #   action = vid_path.split("_")[1]
    #   if action in ["running", "jogging"]:
      n_skip = T
    #   else:
    #     n_skip = T  int(tokens[2])-K-T-1

      for j in xrange(int(tokens[1]),int(tokens[1])+1 ):
        print("Video "+str(i)+"/"+str(len(testfiles))+". Index "+str(j)+
              "/"+str(vid.get_length()-T-1))

        folder_pref = vid_path.split("/")[-1].split(".")[0]
        folder_name = folder_pref+"."+str(j)+"-"+str(j+T)

        vid_names.append(folder_name)
        savedir = "../results/images/KTH/"+prefix+"/"+folder_name

        seq_batch = np.zeros((1,K+T, image_size, image_size,
                         c_dim), dtype="float32")
        diff_batch = np.zeros((1,K-1, image_size, image_size,
                                c_dim), dtype="float32") 
        for t in xrange(K+T):

          # imageio fails randomly sometimes
          while True:
            try:
              img = cv2.resize(vid.get_data(j+t), (image_size, image_size))
              break
            except Exception:
              print("imageio failed loading frames, retrying")

          img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          seq_batch[0,t,:,:] = inverse_transform(transform(img[:,:,None]))

        for t in xrange(1,K):
          prev = (seq_batch[0,t-1,:,:])
          next = (seq_batch[0,t,:,:])
        #   prev = inverse_transform(seq_batch[0,t-1,:,:])
        #   next = inverse_transform(seq_batch[0,t,:,:])
          diff = next.astype("float32")-prev.astype("float32")
          diff_batch[0,t-1,:,:] = diff
        print diff_batch.shape
        accel_batch= np.zeros((1,K-2, image_size,image_size,c_dim),dtype="float32")
        for t in xrange(1,K-1):
          prev_diff= diff_batch[0,t-1,:, :]
          next_diff= diff_batch[0,t,:,:]
          accel_batch[0,t-1,:,:]= next_diff.astype("float32")-prev_diff.astype("float32")
        print accel_batch.shape

        samples = diff_batch[0]
        print samples.shape
        print("Saving velocity_sample ...")
        save_images(samples[:,:,:,::-1], [1, K-1],
                        samples_dir+"velo_inputs_to_network_mod%s.png" % (iters))
                            
        samples = accel_batch[0]
        print samples.shape
        print("Saving accelaration_sample ...")
        save_images(samples[:,:,:,::-1], [1, K-2],
                        samples_dir+"accel_inputs_to_network_mod%s.png" % (iters))
        
        
        true_data = seq_batch[:,K:,:,:,:].copy()
        pred_data = np.zeros(true_data.shape, dtype="float32")
        xt = seq_batch[:,K-1,:,:]
        save_images(xt, [1, 1],
                        samples_dir+"xt_input_to_network_mod%s.png" % (iters))
        pred_data = sess.run([model.G],
                                feed_dict={model.velocity: diff_batch,
                                           model.xt: xt,
                                           model.accelaration:accel_batch})[0]
        print pred_data.shape
        
        # pred_data= pred_data[0]
        # print pred_data.shape
        if not os.path.exists(savedir):
          os.makedirs(savedir)
        sbatch = seq_batch[0, K:, :, :]
        samples = np.concatenate((pred_data[0], sbatch), axis=0)
        print("Saving sample ...")
        save_images(samples[:, :, :, ::-1], [2, T],
                        samples_dir+"test_%s.png" % (14))
                        ##########
    #     cpsnr = np.zeros((K+T,))
    #     cssim = np.zeros((K+T,))
    #     pred_data = np.concatenate((seq_batch[:,:K,:,:], pred_data),axis=1)
    #     true_data = np.concatenate((seq_batch[:,:K,:,:], true_data),axis=1)
    #     for t in xrange(K+T):
    #       pred = (inverse_transform(pred_data[0,t,:,:])*255).astype("float32")    #.astype("uint8")
    #       target = (inverse_transform(true_data[0,t,:,:])*255).astype("float32")         #.astype("uint8")

    #     #   cpsnr[t] = measure.compare_psnr(pred,target, range=[0,1])
    #     #   cssim[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
    #     #                                                cv2.COLOR_GRAY2BGR)),
    #     #                                Image.fromarray(cv2.cvtColor(pred,
    #     #                                                cv2.COLOR_GRAY2BGR)), range=[0,1])
    #     #   pred = draw_frame(pred, t < K)
    #     #   target = draw_frame(target, t < K)

    #       cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
    #       cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

    #     cmd1 = "rm "+savedir+"/pred.gif"
    #     cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
    #             "/pred_%04d.png "+savedir+"/pred.gif")
    #     cmd3 = "rm "+savedir+"/pred*.png"

    #     # Comment out "system(cmd3)" if you want to keep the output images
    #     # Otherwise only the gifs will be kept
    #     #system(cmd1); 
    #     system(cmd2) 
    #     #system(cmd3)

    #     cmd1 = "rm "+savedir+"/gt.gif"
    #     cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
    #             "/gt_%04d.png "+savedir+"/gt.gif")
    #     cmd3 = "rm "+savedir+"/gt*.png"

    #     # Comment out "system(cmd3)" if you want to keep the output images
    #     # Otherwise only the gifs will be kept
    #     #system(cmd1); 
    #     system(cmd2); #system(cmd3)

    #     psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
    #     ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

    # np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
    # print("Results saved to "+save_path)
  print("Done.")

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--prefix", type=str, dest="prefix", required=False, 
                      default= "Paper_model",help="Prefix for log/snapshot")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=128, help="Pre-trained model")
  parser.add_argument("--K", type=int, dest="K",
                      default=10, help="Number of input images")
  parser.add_argument("--T", type=int, dest="T",
                      default=30, help="Number of steps into the future")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=False,
                      default=0,help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))