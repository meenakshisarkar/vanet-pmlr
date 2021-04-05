import os
import cv2
import sys
import time
import ssim
import imageio
import random
import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from vanet import VANET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def write_frames(frames, dir):
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(dir, "{}.png".format(i)), frame)

def main(prefix, image_h, image_w, K, T, vid_type, gpu):
    data_path = "../data/KITTI/"
    f = open(data_path+"train_wo_campus.txt","r")
    testfiles = f.readlines()
    random.shuffle(testfiles)
    n_keep=20
    testfiles = testfiles[:n_keep]
    c_dim = 3
    resize_shape = (image_h, image_w)
    iters = 100
    samples_dir= "../results/images/test_kitti"
    checkpoint_dir="../models/KITTI_Full_VANET_image_h=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.02_lr=0.0001_no_iteration30000_wo_campus/"
    best_model = "VANET.model-29002"
    #   else:
    #     checkpoint_dir = "../models/"+prefix+"/"
    #     best_model = None # will pick last model
    with tf.device("/gpu:0"): #"/gpu:%d"%gpu[0]):
        model = VANET(image_size=[image_h, image_w], c_dim=c_dim,
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

        # quant_dir = "../results/quantitative/KITTI/"+prefix+"/"
        # save_path = quant_dir+"results_model="+model_name+".npz"
        # if not exists(quant_dir):
        #     makedirs(quant_dir)
        base_dir = "../results/network_debug/KITTI/" + prefix + "/"
        # if not exists(save_path):
            # makedirs(save_path)

        vid_names = []
        # psnr_err = np.zeros((0, T))
        # ssim_err = np.zeros((0, T))
        for i in xrange(len(testfiles)):
            vid_dir = testfiles[i].strip()
            vid_path = os.path.join(data_path, vid_dir+"_sync")
            print(vid_path)
            img_files = glob.glob(os.path.join(vid_path, 'image_{}/data/*.png'.format(vid_type)))
            imgs = [imageio.imread(img_file)[np.newaxis, ...] for img_file in sorted(img_files)]
            vid = np.concatenate(imgs, axis=0)
            vid = vid[..., np.newaxis] if vid_type in ['00', '01'] else vid
            low = 0
            high = vid.shape[0] - K - T + 1
            # n_skip = T
            # for j in xrange(low, high, n_skip):
            #     print("Video "+str(i)+"/"+str(len(testfiles))+". Index "+str(j)+
            #             "/"+str(vid.shape[0]-T-1))

            #     folder_pref = vid_dir + "_sync"
            #     folder_name = folder_pref+"."+str(j)+"-"+str(j+T)

            #     vid_names.append(folder_name)
            #     savedir = "../results/images/KITTI/"+prefix+"/"+folder_name
            try:
                j = np.random.randint(low, high)
            except:
                continue

            seq_batch = np.zeros((1,K+T, image_h, image_w, c_dim), dtype="float32")
            diff_batch = np.zeros((1,K-1, image_h, image_w, c_dim), dtype="float32") 
            for t in xrange(K+T):
                img = cv2.resize(vid[j + t], resize_shape[::-1])
                seq_batch[0,t,:,:] = inverse_transform(transform(img))

                # for t in xrange(1,K):
                #     prev = (seq_batch[0,t-1,:,:])
                #     next = (seq_batch[0,t,:,:])
                #     diff = next.astype("float32")-prev.astype("float32")
                #     diff_batch[0,t-1,:,:] = diff
        
                # accel_batch= np.zeros((1,K-2, image_size,image_size,c_dim),dtype="float32")
                # for t in xrange(1,K-1):
                #     prev_diff= diff_batch[0,t-1,:, :]
                #     next_diff= diff_batch[0,t,:,:]
                #     accel_batch[0,t-1,:,:]= next_diff.astype("float32")-prev_diff.astype("float32")
        
            diff_batch = seq_batch[:, 1:K, ...] - seq_batch[:, :K-1, ...]
            accel_batch = diff_batch[:, 1:, ...] - diff_batch[:, :-1, ...]
                # samples = diff_batch[0]
                # print samples.shape
                # print("Saving velocity_sample ...")
                # save_images(samples[:,:,:,::-1], [1, K-1],
                #             samples_dir+"velo_inputs_to_network_mod%s.png" % (iters))
                            
                # samples = accel_batch[0]
                # print samples.shape
                # print("Saving accelaration_sample ...")
                # save_images(samples[:,:,:,::-1], [1, K-2],
                #             samples_dir+"accel_inputs_to_network_mod%s.png" % (iters))
        
        
            true_data = seq_batch[:,K:,:,:,:].copy()
            pred_data = np.zeros(true_data.shape, dtype="float32")
            xt = seq_batch[:,K-1,:,:]
                # save_images(xt, [1, 1],
                #             samples_dir+"xt_input_to_network_mod%s.png" % (iters))
            pred_data = sess.run([model.G], feed_dict={model.velocity: diff_batch,
                                                        model.xt: xt,
                                                        model.accelaration:accel_batch})[0]
            print pred_data.shape
        
            # pred_data= pred_data[0]
            # print pred_data.shape
            veldir = os.path.join(base_dir, vid_dir+"_sync", "vel")
            accdir = os.path.join(base_dir, vid_dir+"_sync", "accel")

            if not exists(veldir):
                os.makedirs(veldir)
            if not exists(accdir):
                os.makedirs(accdir)
            

            write_frames(diff_batch[0]*255, veldir)
            write_frames(accel_batch[0]*255, accdir)
            cv2.imwrite(os.path.join(base_dir, vid_dir + "_sync", "xt.png"), xt[0, :, :]*255)
            cv2.imwrite(os.path.join(base_dir, vid_dir + "_sync", "xtp1.png"), pred_data[0, 0, :, :, :]*255)
                # sbatch = seq_batch[0, K:, :, :]
                # samples = np.concatenate((pred_data[0], sbatch), axis=0)
                # print("Saving sample ...")
                # save_images(samples[:, :, :, ::-1], [2, T],
                #             samples_dir+"test_%s.png" % (14))
                            ##########
                # cpsnr = np.zeros((K+T,))
                # cssim = np.zeros((K+T,))
                # pred_data = np.concatenate((seq_batch[:,:K,:,:], pred_data),axis=1)
                # true_data = np.concatenate((seq_batch[:,:K,:,:], true_data),axis=1)
                # for t in xrange(K+T):
                #     pred = ((pred_data[0,t,:,:])*255).astype("uint8")    #.astype("uint8")
                #     target = ((true_data[0,t,:,:])*255).astype("uint8")         #.astype("uint8")
                #     cpsnr[t] = measure.compare_psnr(pred,target)
                #     cssim[t] = ssim.compute_ssim(Image.fromarray(target), Image.fromarray(pred))
                #     pred = draw_frame(pred, t < K)
                #     target = draw_frame(target, t < K)

                #     cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
                #     cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

                # cmd1 = "rm "+savedir+"/pred.gif"
                # cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                #         "/pred_%04d.png "+savedir+"/pred.gif")
                # cmd3 = "rm "+savedir+"/pred*.png"

                # Comment out "system(cmd3)" if you want to keep the output images
                # Otherwise only the gifs will be kept
                # system(cmd1); 
                # system(cmd2) 
                # system(cmd3)

                # cmd1 = "rm "+savedir+"/gt.gif"
                # cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                #         "/gt_%04d.png "+savedir+"/gt.gif")
                # cmd3 = "rm "+savedir+"/gt*.png"

                # Comment out "system(cmd3)" if you want to keep the output images
                # Otherwise only the gifs will be kept
                # system(cmd1); 
                # system(cmd2); 
                # system(cmd3)

                # psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
                # ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

        # np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        print("Results saved to "+base_dir)
    print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, dest="prefix", required=False, 
                        default= "vanet_wo_gen_v1",help="Prefix for log/snapshot")
    parser.add_argument("--image_h", type=int, dest="image_h",
                        default=64, help="Pre-trained model")
    parser.add_argument("--image_w", type=int, dest="image_w",
                        default=208, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=20, help="Number of steps into the future")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=False,
                        default=0,help="GPU device id")
    parser.add_argument("--vid_type", type=str, dest="vid_type",
                        default='03', help="Grayscale/color, right/left stereo recordings")
    args = parser.parse_args()
    main(**vars(args))
