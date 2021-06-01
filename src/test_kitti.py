# Testing the performance of VANet with KITTI dataset. We measure ssim, psnr and frechet_video_distance between the ground truth image frames and the frames predicted 
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

import numpy as np
from PIL import Image
np.random.seed(77)


def main(lr, batch_size, alpha, beta, image_h, image_w, vid_type, K,
         T, num_iter, gpu, train_gen_only, model_name,iters_start,beta1,train_timesteps,model_no):
    data_path = "../data/KITTI/test"
    test_dirs=[]
    dirs_len=[]
    for d1 in os.listdir(data_path):
        dir_len=int(len(os.listdir(os.path.join(data_path, d1+"/image_03/data"))))
        for l in range(dir_len//40):
            test_dirs.append(os.path.join(data_path, d1+"/image_03/data"+"#"+str(l)))
            dirs_len.append([l*40,l*40+40])
            
    data_dict= dict(zip(test_dirs,dirs_len))
    margin = 0.3

    prefix = ("KITTI_Full-v1_{}".format(model_name)
              + "_GPU_id="+str(gpu)
              + "_image_w="+str(image_w)
              + "_K="+str(K)
              + "_T="+str(train_timesteps)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1)+'_wo_campus')
    prefix_test  = ("KITTI_{}".format(model_name)
              + "_GPU_id="+str(gpu)
              + "_image_w="+str(image_w)
              + "_K="+str(K)
              + "_FutureT="+str(T)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1))

    print("\n"+prefix+"\n")
    checkpoint_dir = "../models/"+prefix+"/"
    samples_dir = "../samples/"+prefix+"/"
    summary_dir = "../logs/"+prefix+"/"
    best_model = model_name+".model-"+model_no
    model_number="model-"+model_no

    with tf.device("/gpu:{}".format(gpu)):
        if model_name == 'VANET':
            model = VANET(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir,training=False)
        else:
            raise ValueError('Model {} undefined'.format(model_name))

        target_vid = tf.placeholder(tf.float32, [16,T,image_h,image_w,3])
        pred_vid = tf.placeholder(tf.float32, [16,T,image_h,image_w,3]) 

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
        quant_dir = "../results/quantitative/KITTI/"+prefix_test+"/"+model_number+"/"
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
            seq_batch, diff_batch, accel_batch = load_kitti_data(d, l,(image_h, image_w), K, T)
            seq_batch = seq_batch[None, ...]
            diff_batch = diff_batch[None, ...]
            accel_batch = accel_batch[None, ...]
            #print(seq_batch.shape)
            true_data = seq_batch[:,K:,:,:,:].copy()
            pred_data = np.zeros(true_data.shape, dtype="float32")
            xt = seq_batch[:,K-1,:,:,:]
            # save_images(xt, [1, 1],
            #             samples_dir+"xt_input_to_network_mod%s.png" % (iters))
            if model_name == 'VANET':
                pred_data = sess.run([model.G],
                                        feed_dict={model.velocity: diff_batch, model.xt: xt, model.accelaration:accel_batch})[0]

            # print (pred_data.shape)
            savedir = os.path.join('../results/images/KITTI/'+prefix_test+'/'+model_number,'/'.join(d.split('/')[-4:]))
            print (savedir )

            if not os.path.exists(savedir):
                os.makedirs(savedir)
            
            cpsnr = np.zeros((K+T,))
            cssim = np.zeros((K+T,))

            true_data_lst.append((true_data*255).astype("uint8"))
            pred_data_lst.append((pred_data*255).astype("uint8"))
            pred_data = np.concatenate((seq_batch[:,:K,:,:,:], pred_data),axis=1)
            true_data = np.concatenate((seq_batch[:,:K,:,:,:], true_data),axis=1)
            for t in range(K+T):
                pred = ((pred_data[0,t,:,:,:])*255).astype("uint8")    #.astype("uint8")
                target = ((true_data[0,t,:,:,:])*255).astype("uint8")         #.astype("uint8")
                cpsnr[t] = metrics.peak_signal_noise_ratio(pred,target)
                # cssim[t] = ssim.compute_ssim(Image.fromarray(target), Image.fromarray(pred))
                cssim[t] = metrics.structural_similarity(target, pred, multichannel=True)
                # images=np.concatenate((target[None,:,:,:],pred[None,:,:,:]),axis=0)
                pred = draw_frame(pred, t < K)
                target = draw_frame(target, t < K)

                cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
                cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

            cmd1 = "rm "+savedir+"/pred.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                    "/pred_%04d.png "+savedir+"/pred.gif")
            cmd3 = "rm "+savedir+"/pred*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); 
            system(cmd2) 
            # system(cmd3)

            cmd1 = "rm "+savedir+"/gt.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                    "/gt_%04d.png "+savedir+"/gt.gif")
            cmd3 = "rm "+savedir+"/gt*.png"

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
                        default=0.0001, help="GAN loss weight")
    parser.add_argument("--image_h", type=int, dest="image_h",
                        default=64, help="Frame height")
    parser.add_argument("--image_w", type=int, dest="image_w",
                        # default=208, help="Frame width")
                        default=64, help="Frame width")
    parser.add_argument("--vid_type", type=str, dest="vid_type",
                        default='03', help="Grayscale/color, right/left stereo recordings")
    parser.add_argument("--model_name", type=str, dest="model_name",
                        default='VANET', help="model to train vanet/vnet")
    parser.add_argument("--model_no", type=str, dest="model_no",
                        default='150000', help="modelnumber from checkpoint for best performance")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=20, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=150000, help="Number of iterations")
    parser.add_argument("--gpu", type=int,  dest="gpu", required=False,
                        default=0, help="GPU device id")
    parser.add_argument("--beta1", type=float,  dest="beta1", required=False,
                        default=0.9, help="beta1 decay rate")
    parser.add_argument("--train_timesteps", type=int,  dest="train_timesteps", required=False,
                        default=10, help="future time steps")
    parser.add_argument("--train_gen_only", default=False, action='store_true')
    parser.add_argument("--iters_start", type=int,  dest="iters_start", required=False, default=0, help='iteration_starts')

    args = parser.parse_args()
    main(**vars(args))


