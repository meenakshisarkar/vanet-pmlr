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
# from vnet import VNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(prefix, image_h, image_w, K, T, gpu):
    data_path = "../data/BAIR/processed_data/test"
    test_dirs = []
    for d1 in os.listdir(data_path):
        for d2 in os.listdir(os.path.join(data_path, d1)):
            test_dirs.append(os.path.join(data_path, d1, d2))
    c_dim = 3
    resize_shape = (image_h, image_w)
    iters = 100
    samples_dir= "../results/images/test_bair/"
    # checkpoint_dir="../models/BAIR_Full_VANET_GPU_id=1_image_h=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.001_lr=0.0001_no_iteration150000/"
    # best_model = "VANET.model-151000"
    model_spec="BAIR_Full_VANET_GPU_id=0_image_h=64_K=10_T=10_batch_size=16_alpha=1.0_beta=0.001_lr=0.0001_no_iteration150000"
    checkpoint_dir="../models/"+model_spec+"/"
    best_model = "VANET.model-151000"

    #   else:
    #     checkpoint_dir = "../models/"+prefix+"/"
    #     best_model = None # will pick last model
    with tf.device("/gpu:{}".format(gpu)): #"/gpu:%d"%gpu[0]):
        #model = VANET(image_size=[image_h, image_w], c_dim=c_dim,
        #timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir, training=False)
        model = VANET(image_size=[image_h, image_w], c_dim = c_dim,
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

        quant_dir = "../results/quantitative/BAIR/"+prefix+"/"+model_spec+"/"
        save_path = quant_dir+"results_model="+model_name+".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)


        vid_names = []
        psnr_err = np.zeros((0, T))
        ssim_err = np.zeros((0, T))
        for i in range(len(test_dirs)):
            
            d = test_dirs[i]
            seq_batch, diff_batch, accel_batch = load_bair_data(d, K, T)
            seq_batch = seq_batch[None, ...]
            diff_batch = diff_batch[None, ...]
            accel_batch = accel_batch[None, ...]
            #print(seq_batch.shape)
            true_data = seq_batch[:,K:,:,:,:].copy()
            pred_data = np.zeros(true_data.shape, dtype="float32")
            xt = seq_batch[:,K-1,:,:,:]
            # save_images(xt, [1, 1],
            #             samples_dir+"xt_input_to_network_mod%s.png" % (iters))
            pred_data = sess.run([model.G],
                                    feed_dict={model.velocity: diff_batch, model.xt: xt, model.accelaration:accel_batch})[0]
            print (pred_data.shape)
            savedir = os.path.join('../results/images/BAIR/'+model_spec,'/'.join(d.split('/')[-3:]))
            print (savedir )
        # pred_data= pred_data[0]
        # print pred_data.shape
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            # sbatch = seq_batch[0, K:, :, :]
            # samples = np.concatenate((pred_data[0], sbatch), axis=0)
            # print("Saving sample ...")
            # save_images(samples[:, :, :, ::-1], [2, T],
            #             samples_dir+"test_%s.png" % (14))
                        ##########
            cpsnr = np.zeros((K+T,))
            cssim = np.zeros((K+T,))
            pred_data = np.concatenate((seq_batch[:,:K,:,:,:], pred_data),axis=1)
            true_data = np.concatenate((seq_batch[:,:K,:,:,:], true_data),axis=1)
            for t in range(K+T):
                pred = ((pred_data[0,t,:,:,:])*255).astype("uint8")    #.astype("uint8")
                target = ((true_data[0,t,:,:,:])*255).astype("uint8")         #.astype("uint8")
                cpsnr[t] = metrics.peak_signal_noise_ratio(pred,target)
                # cssim[t] = ssim.compute_ssim(Image.fromarray(target), Image.fromarray(pred))
                # cssim[t] = metrics.structural_similarity(target, pred)
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
            system(cmd3)

            cmd1 = "rm "+savedir+"/gt.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                    "/gt_%04d.png "+savedir+"/gt.gif")
            cmd3 = "rm "+savedir+"/gt*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); 
            system(cmd2); 
            system(cmd3)

            psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

        # np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        np.savez(save_path, psnr=psnr_err)
        print("Results saved to "+save_path)

    print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, dest="prefix", required=False, 
                        default= "vanet_wo_gen_v1",help="Prefix for log/snapshot")
    parser.add_argument("--image_h", type=int, dest="image_h",
                        default=64, help="Pre-trained model")
    parser.add_argument("--image_w", type=int, dest="image_w",
                        default=64, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=20, help="Number of steps into the future")
    parser.add_argument("--gpu", type=int, dest="gpu", required=False,
                        default=0,help="GPU device id")
    args = parser.parse_args()
    main(**vars(args))
