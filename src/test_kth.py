# Testing the performance of VANet with KTH dataset. We measure ssim, psnr and frechet_video_distance between the ground truth image frames and the frames predicted 
# by VANet and the resultant arrays are saved in .npz file.


import os
import cv2
import sys
import time
import ssim
import imageio

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.random.set_random_seed(77)
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure
import skimage.metrics as metrics

from vanet import VANET
from vanet_ntd import VANET_ntd
from vnet import VNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw
import frechet_video_distance as fvd
np.random.seed(77)


def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, train_gen_only, model_name,iters_start,beta1,train_timesteps,model_no):
  data_path = "../data/KTH/processed/test"
  train_txt_path = "../data/KTH/"
  f = open(train_txt_path+"test_kth_trimmed.txt","r")
  trainfiles = f.readlines()
  dir_counter=range(len(trainfiles))
  train_dirs=[]
  dirs_len=[]
  for dir_index, lines in zip(dir_counter, trainfiles):
    d1=lines.split('_')
    d2=d1[-1].split()
    if int(d2[2]) -int(d2[1]) < (T+K):
      break
    else:
      train_dirs.append(os.path.join(data_path, d1[1],d1[0]+'-'+d2[0]+'#'+str(dir_index)))
      dirs_len.append([d2[1],d2[2]])
  data_dict= dict(zip(train_dirs,dirs_len))
  prefix  = ("KTH_{}".format(model_name)+'_NoGAN'
              + "_GPU_id="+str(gpu)
              + "_image_w="+str(image_size)
              + "_K="+str(K)
              + "_T="+str(train_timesteps)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1))
  prefix_test  = ("KTH_{}".format(model_name)+'_NoGAN'
              + "_GPU_id="+str(gpu)
              + "_image_w="+str(image_size)
              + "_K="+str(K)
              + "_FutureT="+str(T)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../models/"+prefix+"/"
  # best_model = "VNET.model-26002"
  samples_dir = "../samples/"+prefix+"/"
  summary_dir = "../logs/"+prefix+"/"
  best_model = model_name+".model-"+model_no
  model_number="model-"+model_no
#   else:
#     checkpoint_dir = "../models/"+prefix+"/"
#     best_model = None # will pick last model

  with tf.device("/gpu:{}".format(gpu)):
        if model_name == 'VANET':
            model = VANET(image_size=[image_size, image_size], c_dim=1,
                timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir,training=False)
        #elif model_name == add other model names to generate comperative performance with VANET
            
        else:
            raise ValueError('Model {} undefined'.format(model_name))

        target_vid = tf.placeholder(tf.float32, [16,T,image_size,image_size,3])
        pred_vid = tf.placeholder(tf.float32, [16,T,image_size,image_size,3]) 

        fvd_err=fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(target_vid,
                                                    (224, 224))),
            fvd.create_id3_embedding(fvd.preprocess(pred_vid,
                                                    (224, 224))))
    
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False,
                                            gpu_options=None)) as sess:  #add gpu_option

    tf.global_variables_initializer().run()
    sess.run(tf.tables_initializer())
    success_load_model = model.load(sess, checkpoint_dir,best_model)
        # print(success_load_model[0])

    if success_load_model[0]:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
        return
    quant_dir = "../results/quantitative/KTH/"+prefix_test+"/"+model_number+"/"
    save_path = quant_dir+"results_model="+model_name+".npz"
    if not exists(quant_dir):
      makedirs(quant_dir)

    vid_names = []
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))
    true_data_lst=[]
    pred_data_lst=[]
    fvd_score=[]
    for d, l in data_dict.items():
        
        # d = test_dirs[i]
        seq_batch, diff_batch, accel_batch = load_kth_data(d,l,(image_size, image_size), K, T)
        seq_batch = seq_batch[None, ...]
        diff_batch = diff_batch[None, ...]
        accel_batch = accel_batch[None, ...]
        #print(seq_batch.shape)
        true_data = seq_batch[:,K:,:,:,:].copy()
        pred_data = np.zeros(true_data.shape, dtype="float32")
        xt = seq_batch[:,K-1,:,:,:]
        # save_images(xt, [1, 1],
        #             samples_dir+"xt_input_to_network_mod%s.png" % (iters))
        if model_name=='VANET':
            pred_data = sess.run([model.G],
                                    feed_dict={model.velocity: diff_batch, model.xt: xt, model.accelaration:accel_batch})[0]
        else:
            print("error!!!!")
            return

        savedir = os.path.join('../results/images/KTH/'+prefix_test+'/'+model_number,'/'.join(d.split('/')[-3:]))
        print (savedir )

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        cpsnr = np.zeros((K+T,))
        cssim = np.zeros((K+T,))

        pred_data = np.concatenate((seq_batch[:,:K,:,:,:], pred_data),axis=1)
        true_data = np.concatenate((seq_batch[:,:K,:,:,:], true_data),axis=1)
        true_dataRGB=np.concatenate((true_data,true_data,true_data), axis=-1)
        pred_dataRGB=np.concatenate((pred_data,pred_data,pred_data), axis=-1)
        true_data_lst.append((true_dataRGB[:,K:,:,:,:]*255).astype("uint8"))
        pred_data_lst.append((pred_dataRGB[:,K:,:,:,:]*255).astype("uint8"))
        for t in range(K+T):
            pred = ((pred_data[0,t,:,:,:])*255).astype("uint8")    #.astype("uint8")
            target = ((true_data[0,t,:,:,:])*255).astype("uint8")         #.astype("uint8")
            pred_RGB= ((pred_dataRGB[0,t,:,:,:])*255).astype("uint8")    #.astype("uint8")
            target_RGB = ((true_dataRGB[0,t,:,:,:])*255).astype("uint8")         #.astype("uint8")
            cpsnr[t] = metrics.peak_signal_noise_ratio(pred,target)
            # cssim[t] = ssim.compute_ssim(Image.fromarray(target), Image.fromarray(pred))
            cssim[t] = metrics.structural_similarity(target, pred, multichannel=True)
            images=np.concatenate((target_RGB[None,...],pred_RGB[None,...]),axis=0)
            
            pred = draw_frame(pred, t < K)
            target = draw_frame(target, t < K)

            cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
            cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

        cmd1 = "rm "+savedir+"/pred.gif"
        cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                "/pred_%04d.png "+savedir+"/pred.gif")
        cmd3 = "rm "+savedir+"/pred*.png"

        system(cmd1); 
        system(cmd2) 
        # system(cmd3)

        cmd1 = "rm "+savedir+"/gt.gif"
        cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                "/gt_%04d.png "+savedir+"/gt.gif")
        cmd3 = "rm "+savedir+"/gt*.png"

        # Comment out "system(cmd3)" if you want to keep the output images
        # Otherwise only the gifs will be kept
        system(cmd1); 
        system(cmd2); 
        # system(cmd3)

        psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
        ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

    true_data_lst= np.concatenate(true_data_lst, axis=0)
    pred_data_lst=np.concatenate(pred_data_lst, axis=0)
    for i in range(0, len(list(data_dict.keys()))//16):
            fvd_score.append(sess.run(fvd_err, feed_dict={target_vid: np.squeeze(true_data_lst[i*16:i*16+16,...]),
                                 pred_vid: np.squeeze(pred_data_lst[i*16:i*16+16,...])}))

    print("fvd: "+str(fvd_score))
        # return
    fvd_score= np.array(fvd_score)
    np.savez(save_path, psnr=psnr_err, ssim=ssim_err, fvd_score=fvd_score )
    print("Results saved to "+save_path)

  print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.001, help="GAN loss weight")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=64, help="Frame height")
    parser.add_argument("--model_name", type=str, dest="model_name",
                        default='VANET', help="model to train vanet/vnet")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=20, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=150000, help="Number of iterations")
    parser.add_argument("--gpu", type=int,  dest="gpu", required=False,
                        default=1, help="GPU device id")
    parser.add_argument("--beta1", type=float,  dest="beta1", required=False,
                        default=0.9, help="beta1 decay rate")
    parser.add_argument("--train_gen_only", default=False, action='store_true')
    parser.add_argument("--iters_start", type=int,  dest="iters_start", required=False, default=0, help='iteration_starts')
    parser.add_argument("--train_timesteps", type=int,  dest="train_timesteps", required=False,
                          default=10, help="future time steps")
    parser.add_argument("--model_no", type=str, dest="model_no",
                        default='150000', help="modelnumber from checkpoint for best performance")
    args = parser.parse_args()
    main(**vars(args))

