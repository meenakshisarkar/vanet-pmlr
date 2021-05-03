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
np.random.seed(77)


def main(lr, batch_size, alpha, beta, image_h, image_w, vid_type, K,
         T, num_iter, gpu, train_gen_only, model_name,iters_start,beta1,train_timesteps,model_no):
    data_path = "../data/KITTI/test2"
    test_dirs=[]
    dirs_len=[]
    for d1 in os.listdir(data_path):
        dir_len=int(len(os.listdir(os.path.join(data_path, d1+"/image_03/data"))))
        for l in range(dir_len//40):
            test_dirs.append(os.path.join(data_path, d1+"/image_03/data"+"#"+str(l)))
            dirs_len.append([l*40,l*40+40])
            # dirs_len.append(len(os.listdir(os.path.join(data_path, d1+"/image_03/data"+"#"+str(l)))))

        # dirs_len.append(len(os.listdir(os.path.join(data_path, d1+"/image_03/data"))))
        # test_dirs.append(os.path.join(data_path, d1+"/image_03/data"))
    # with open(data_path+"train_wo_campus.txt", "r") as f:
    #     trainfiles = f.readlines()
    data_dict= dict(zip(test_dirs,dirs_len))
    margin = 0.3
    # updateD = True
    # updateG = True
    # iters = iters_start
    prefix = ("KITTI_Full_{}".format(model_name)
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
    # best_model = "VNET.model-26002"
    samples_dir = "../samples/"+prefix+"/"
    summary_dir = "../logs/"+prefix+"/"
    best_model = model_name+".model-"+model_no
    model_number="model-"+model_no

    with tf.device("/gpu:{}".format(gpu)):
        if model_name == 'VANET':
            model = VANET(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir,training=False)
        elif model_name == 'VANET_ntd':
            model = VANET_ntd(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir,training=False)
        elif model_name == 'VNET':
            model = VNET(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir,training=False)
        else:
            raise ValueError('Model {} undefined'.format(model_name))
    gpu_options = tf.GPUOptions(allow_growth=True)
    # (config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options if gpus else None))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:                                     #if gpus else None

        tf.global_variables_initializer().run()

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
            elif model_name == 'VNET':
                pred_data = sess.run([model.G],
                                    feed_dict={model.velocity: diff_batch, model.xt: xt})[0]
            elif model_name == 'VANET_ntd': 
                pred_data = sess.run([model.G],
                                    feed_dict={model.velocity: diff_batch, model.xt: xt, model.accelaration:accel_batch})[0]

            print (pred_data.shape)
            savedir = os.path.join('../results/images/KITTI/'+prefix_test+'/'+model_number,'/'.join(d.split('/')[-4:]))
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
                cssim[t] = metrics.structural_similarity(target, pred, multichannel=True)
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

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); 
            system(cmd2); 
            # system(cmd3)

            psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

        np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        # np.savez(save_path, psnr=psnr_err)
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

# def main(prefix, image_h, image_w, K, T, vid_type, gpu):
#     data_path = "../data/KITTI/"
#     f = open(data_path+"test_wo_campus.txt","r")
#     testfiles = f.readlines()
#     c_dim = 3
#     resize_shape = (image_h, image_w)
#     iters = 100
#     samples_dir= "../results/images/test_kitti"
#     checkpoint_dir="../models/KITTI_Full_VNET_image_h=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.02_lr=0.0001_no_iteration30000_wo_campus/"
#     best_model = "VNET.model-26002"
#     #   else:
#     #     checkpoint_dir = "../models/"+prefix+"/"
#     #     best_model = None # will pick last model
#     with tf.device("/gpu:{}".format(gpu)): #"/gpu:%d"%gpu[0]):
#         #model = VANET(image_size=[image_h, image_w], c_dim=c_dim,
#         #timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir, training=False)
#         model = VNET(image_size=[image_h, image_w], c_dim = c_dim,
#         timesteps=K, batch_size=1, F=T, checkpoint_dir=checkpoint_dir, training=False)  
    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
#                                             log_device_placement=False,
#                                             gpu_options=None)) as sess:  #add gpu_option

#         tf.global_variables_initializer().run()

#         loaded, model_name = model.load(sess, checkpoint_dir,best_model)

#         if loaded:
#             print(" [*] Load SUCCESS")
#         else:
#             print(" [!] Load failed... exitting")
#             return

#         quant_dir = "../results/quantitative/KITTI/"+prefix+"/"
#         save_path = quant_dir+"results_model="+model_name+".npz"
#         if not exists(quant_dir):
#             makedirs(quant_dir)


#         vid_names = []
#         psnr_err = np.zeros((0, T))
#         ssim_err = np.zeros((0, T))
#         for i in xrange(len(testfiles)):
#             vid_dir = testfiles[i].strip()
#             vid_path = os.path.join(data_path, vid_dir+"_sync")
#             print(vid_path)
#             img_files = glob.glob(os.path.join(vid_path, 'image_{}/data/*.png'.format(vid_type)))
#             imgs = [imageio.imread(img_file)[np.newaxis, ...] for img_file in sorted(img_files)]
#             vid = np.concatenate(imgs, axis=0)
#             vid = vid[..., np.newaxis] if vid_type in ['00', '01'] else vid
#             low = 0
#             high = vid.shape[0] - K - T + 1
#             n_skip = T
#             for j in xrange(low, high, n_skip):
#                 print("Video "+str(i)+"/"+str(len(testfiles))+". Index "+str(j)+
#                         "/"+str(vid.shape[0]-T-1))

#                 folder_pref = vid_dir + "_sync"
#                 folder_name = folder_pref+"."+str(j)+"-"+str(j+T)

#                 vid_names.append(folder_name)
#                 savedir = "../results/images/KITTI/"+prefix+"/"+folder_name

#                 seq_batch = np.zeros((1,K+T, image_h, image_w,
#                                         c_dim), dtype="float32")
#                 diff_batch = np.zeros((1,K-1, image_h, image_w,
#                                         c_dim), dtype="float32") 
#                 for t in xrange(K+T):
#                     img = cv2.resize(vid[j + t], resize_shape[::-1])
#                     seq_batch[0,t,:,:] = inverse_transform(transform(img))

#                 # for t in xrange(1,K):
#                 #     prev = (seq_batch[0,t-1,:,:])
#                 #     next = (seq_batch[0,t,:,:])
#                 #     diff = next.astype("float32")-prev.astype("float32")
#                 #     diff_batch[0,t-1,:,:] = diff
        
#                 # accel_batch= np.zeros((1,K-2, image_size,image_size,c_dim),dtype="float32")
#                 # for t in xrange(1,K-1):
#                 #     prev_diff= diff_batch[0,t-1,:, :]
#                 #     next_diff= diff_batch[0,t,:,:]
#                 #     accel_batch[0,t-1,:,:]= next_diff.astype("float32")-prev_diff.astype("float32")
        
#                 diff_batch = seq_batch[:, 1:K, ...] - seq_batch[:, :K-1, ...]
#                 accel_batch = diff_batch[:, 1:, ...] - diff_batch[:, :-1, ...]
#                 # samples = diff_batch[0]
#                 # print samples.shape
#                 # print("Saving velocity_sample ...")
#                 # save_images(samples[:,:,:,::-1], [1, K-1],
#                 #             samples_dir+"velo_inputs_to_network_mod%s.png" % (iters))
                            
#                 # samples = accel_batch[0]
#                 # print samples.shape
#                 # print("Saving accelaration_sample ...")
#                 # save_images(samples[:,:,:,::-1], [1, K-2],
#                 #             samples_dir+"accel_inputs_to_network_mod%s.png" % (iters))
        
        
#                 true_data = seq_batch[:,K:,:,:,:].copy()
#                 pred_data = np.zeros(true_data.shape, dtype="float32")
#                 xt = seq_batch[:,K-1,:,:,:]
#                 # save_images(xt, [1, 1],
#                 #             samples_dir+"xt_input_to_network_mod%s.png" % (iters))
#                 pred_data = sess.run([model.G],
#                                         feed_dict={model.velocity: diff_batch,
#                                                     model.xt: xt})[0]
#                                                     #model.accelaration:accel_batch})[0]
#                 print pred_data.shape
        
#             # pred_data= pred_data[0]
#             # print pred_data.shape
#                 if not os.path.exists(savedir):
#                     os.makedirs(savedir)
#                 # sbatch = seq_batch[0, K:, :, :]
#                 # samples = np.concatenate((pred_data[0], sbatch), axis=0)
#                 # print("Saving sample ...")
#                 # save_images(samples[:, :, :, ::-1], [2, T],
#                 #             samples_dir+"test_%s.png" % (14))
#                             ##########
#                 cpsnr = np.zeros((K+T,))
#                 cssim = np.zeros((K+T,))
#                 pred_data = np.concatenate((seq_batch[:,:K,:,:,:], pred_data),axis=1)
#                 true_data = np.concatenate((seq_batch[:,:K,:,:,:], true_data),axis=1)
#                 for t in xrange(K+T):
#                     pred = ((pred_data[0,t,:,:,:])*255).astype("uint8")    #.astype("uint8")
#                     target = ((true_data[0,t,:,:,:])*255).astype("uint8")         #.astype("uint8")
#                     cpsnr[t] = measure.compare_psnr(pred,target)
#                     cssim[t] = ssim.compute_ssim(Image.fromarray(target), Image.fromarray(pred))
#                     pred = draw_frame(pred, t < K)
#                     target = draw_frame(target, t < K)

#                     cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
#                     cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

#                 cmd1 = "rm "+savedir+"/pred.gif"
#                 cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
#                         "/pred_%04d.png "+savedir+"/pred.gif")
#                 cmd3 = "rm "+savedir+"/pred*.png"

#                 # Comment out "system(cmd3)" if you want to keep the output images
#                 # Otherwise only the gifs will be kept
#                 system(cmd1); 
#                 system(cmd2) 
#                 system(cmd3)

#                 cmd1 = "rm "+savedir+"/gt.gif"
#                 cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
#                         "/gt_%04d.png "+savedir+"/gt.gif")
#                 cmd3 = "rm "+savedir+"/gt*.png"

#                 # Comment out "system(cmd3)" if you want to keep the output images
#                 # Otherwise only the gifs will be kept
#                 system(cmd1); 
#                 system(cmd2); 
#                 system(cmd3)

#                 psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
#                 ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

#         np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
#         print("Results saved to "+save_path)
#     print("Done.")


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--prefix", type=str, dest="prefix", required=False, 
#                         default= "vanet_wo_gen_v1",help="Prefix for log/snapshot")
#     parser.add_argument("--image_h", type=int, dest="image_h",
#                         default=64, help="Pre-trained model")
#     parser.add_argument("--image_w", type=int, dest="image_w",
#                         default=208, help="Pre-trained model")
#     parser.add_argument("--K", type=int, dest="K",
#                         default=10, help="Number of input images")
#     parser.add_argument("--T", type=int, dest="T",
#                         default=20, help="Number of steps into the future")
#     parser.add_argument("--gpu", type=int, dest="gpu", required=False,
#                         default=0,help="GPU device id")
#     parser.add_argument("--vid_type", type=str, dest="vid_type",
#                         default='03', help="Grayscale/color, right/left stereo recordings")
#     args = parser.parse_args()
#     main(**vars(args))
