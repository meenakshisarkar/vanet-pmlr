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
# from vgg16_feature import *
from sklearn.metrics.pairwise import cosine_similarity
import frechet_video_distance as fvd
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
np.random.seed(77)


def main(lr, batch_size, alpha, beta, image_h, image_w, K,
         T, num_iter, gpu, model_name,beta1,train_timesteps,model_no):
    # data_path = "../data/BAIR/processed_data/test"
    data_path = "../data/BAIR/processed_data_towel_pick/test"
    # data_path = "../temp/processed_data_towel_pick/test"
    c_dim = 3
    resize_shape = (image_h, image_w)
    iters = 100
    test_dirs = []
    for d1 in os.listdir(data_path):
        for d2 in os.listdir(os.path.join(data_path, d1)):
            test_dirs.append(os.path.join(data_path, d1, d2))
    
    # prefix = ("BAIR_Full_{}".format(model_name)
    prefix = ("BAIR_Towel_{}".format(model_name)
              + "_GPU_id="+str(gpu)
              + "_image_h="+str(image_h)
              + "_K="+str(K)
              + "_T="+str(train_timesteps)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration"+str(num_iter)
              +"_beta1"+str(beta1))
    prefix_test  = ("BAIR_towel_{}".format(model_name)
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
    # checkpoint_dir = "../temp/"+prefix+"/"
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
        model_vgg = VGG16(weights='imagenet', include_top=False)
        


        success_load_model = model.load(sess, checkpoint_dir,best_model)
        # print(success_load_model[0])

        if success_load_model[0]:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed... exitting")
            return

        quant_dir = "../results/quantitative/BAIR/"+prefix_test+"/"+model_number+"/"
        save_path = quant_dir+"results_model="+model_name+".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)


        vid_names = []
        psnr_err = np.zeros((0, T))
        ssim_err = np.zeros((0, T))
        vgg16_csim_err=np.zeros((0, T))
        true_data_lst=[]
        pred_data_lst=[]
        fvd_score=[]
        for i in range(len(test_dirs)):
            
            d = test_dirs[i]
            seq_batch, diff_batch, accel_batch = load_bair_towel_data(d, K, T)
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
            # pred_data = sess.run([model.G],
                                    # feed_dict={model.velocity: diff_batch, model.xt: xt, model.accelaration:accel_batch})[0]
            print (pred_data.shape)
            savedir = os.path.join('../results/images/BAIR/'+prefix_test+'/'+model_number,'/'.join(d.split('/')[-3:]))
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
            vgg_csim=np.zeros((K+T,))
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
                images=np.concatenate((target[None,:,:,:],pred[None,:,:,:]),axis=0)
                vgg16_feature_list=[]
                for i in [0,1]:
            # img = image.load_img(np.squeeze(images[i,:,:,:]), target_size=(224, 224))
                    img = Image.fromarray(images[i,:,:,:]).resize((224, 224))
                    # img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    vgg16_feature = model_vgg.predict(img)
                    vgg16_feature_np = np.array(vgg16_feature)
                    vgg16_feature_list.append(vgg16_feature_np.flatten())
                vgg16_ft_lst=np.array(vgg16_feature_list)



                # vgg16_ft_lst=vgg16_feature(np.concatenate((target[None,:,:,:],pred[None,:,:,:]),axis=0), channel=3)
                vgg_csim[t]=cosine_similarity(vgg16_ft_lst[None,0,:],vgg16_ft_lst[None, 1,:])
                pred = draw_frame(pred, t < K)
                target = draw_frame(target, t < K)

                cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
                cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

            cmd1 = "rm "+savedir+"/pred.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                    "/pred_%04d.png "+savedir+"/pred.gif")
            # cmd3 = "rm "+savedir+"/pred*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); 
            system(cmd2) 
            # system(cmd3)

            cmd1 = "rm "+savedir+"/gt.gif"
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                    "/gt_%04d.png "+savedir+"/gt.gif")
            # cmd3 = "rm "+savedir+"/gt*.png"

            # Comment out "system(cmd3)" if you want to keep the output images
            # Otherwise only the gifs will be kept
            system(cmd1); 
            system(cmd2); 
            # system(cmd3)

            psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)
            vgg16_csim_err = np.concatenate((vgg16_csim_err, vgg_csim[None,K:]), axis=0)

        true_data_lst= np.concatenate(true_data_lst, axis=0)
        pred_data_lst=np.concatenate(pred_data_lst, axis=0)
        for i in range(0, len(test_dirs)//16):
            fvd_score.append(sess.run(fvd_err, feed_dict={target_vid: np.squeeze(true_data_lst[i*16:i*16+16,...]),
                                 pred_vid: np.squeeze(pred_data_lst[i*16:i*16+16,...])}))

        print("fvd: "+str(fvd_score))
            # return
        fvd_score= np.array(fvd_score)
        np.savez(save_path, psnr=psnr_err, ssim=ssim_err,vgg16_csim=vgg16_csim_err, fvd_score=fvd_score )

        # np.savez(save_path, psnr=psnr_err, ssim=ssim_err,vgg16_csim=vgg16_csim_err )
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
                        default=64, help="Frame width")
    parser.add_argument("--model_name", type=str, dest="model_name",
                        default='VANET', help="model to train vanet/vnet")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=250000, help="Number of iterations")
    parser.add_argument("--gpu", type=int,  dest="gpu", required=False,
                        default=0, help="GPU device id")
    parser.add_argument("--beta1", type=float,  dest="beta1", required=False,
                        default=0.5, help="beta1 decay rate")
    parser.add_argument("--train_timesteps", type=int,  dest="train_timesteps", required=False,
                        default=10, help="future time steps")
    parser.add_argument("--model_no", type=str, dest="model_no",
                        default='200000', help="modelnumber from checkpoint for best performance")
    

    args = parser.parse_args()
    main(**vars(args))
