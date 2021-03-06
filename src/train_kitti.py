# Training on KITTI dataset.

import cv2
import sys
import time
import imageio

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.random.set_random_seed(77)
import scipy.misc as sm
import numpy as np
import scipy.io as sio
import os

from vanet import VANET
from vanet_ntd import VANET_ntd
from vnet import VNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main(lr, batch_size, alpha, beta, image_h, image_w, vid_type, K,
         T, num_iter, gpu, train_gen_only, model_name,iters_start,beta1):
    data_path = "../data/KITTI/train"
    train_dirs=[]
    dirs_len=[]
    for d1 in os.listdir(data_path):
        dir_len=int(len(os.listdir(os.path.join(data_path, d1+"/image_03/data"))))
        for l in range(dir_len//30):
            train_dirs.append(os.path.join(data_path, d1+"/image_03/data"+"#"+str(l)))
            dirs_len.append([l*30,l*30+30])

    data_dict= dict(zip(train_dirs,dirs_len))
    margin = 0.3
    updateD = True
    updateG = True
    iters = iters_start
    prefix = ("KITTI_Full-v1_{}".format(model_name)
              + "_GPU_id="+str(gpu)
              + "_image_w="+str(image_w)
              + "_K="+str(K)
              + "_T="+str(T)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1)+'_wo_campus')

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


    with tf.device("/gpu:{}".format(gpu)):
        if model_name == 'VANET':
            model = VANET(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=batch_size, F=T, checkpoint_dir=checkpoint_dir)
        elif model_name == 'VANET_ntd':
            model = VANET_ntd(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=batch_size, F=T, checkpoint_dir=checkpoint_dir)
        elif model_name == 'VNET':
            model = VNET(image_size=[image_h, image_w], c_dim=3,
                timesteps=K, batch_size=batch_size, F=T, checkpoint_dir=checkpoint_dir)
        else:
            raise ValueError('Model {} undefined'.format(model_name))

        if train_gen_only:
            g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(model.reconst_loss, var_list=model.g_vars) 
        
        else:
            d_optim, g_optim = (
                tf.train.AdamOptimizer(lr, beta1=0.5).minimize(model.d_loss, var_list=model.d_vars), 
                tf.train.AdamOptimizer(lr, beta1=0.5).minimize(alpha*model.reconst_loss+beta*model.L_gen, var_list=model.g_vars)
                )

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    gpu_options = tf.GPUOptions(allow_growth=True)
    # (config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options if gpus else None))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:                                     #if gpus else None

        tf.global_variables_initializer().run()

        success_load_model = model.load(sess, checkpoint_dir)
        print(success_load_model[0])

        if success_load_model[0]:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        if train_gen_only:
            g_sum = tf.summary.merge([model.L_p_sum,
                        model.L_stgdl_sum, model.L_sum])        
        else:
            g_sum = tf.summary.merge([model.L_p_sum,
                        model.L_stgdl_sum, model.L_sum,
                        model.L_Gen_sum])
            d_sum = tf.summary.merge([model.d_loss_real_sum,
                        model.d_loss_fake_sum, model.d_loss_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters+1
        start_time = time.time()
        rem_ip_samples, rem_gen_samples, rem_model = [], [], []
        with Parallel(n_jobs=batch_size) as parallel:
            while iters < num_iter:
                t0 = time.time()
                mini_batches = get_minibatches_idx(
                    # len(trainfiles), batch_size, shuffle=True)
                    len(train_dirs), batch_size, shuffle=True)
                for _, batchidx in mini_batches:
                    if len(batchidx) == batch_size:
                        seq_batch = np.zeros((batch_size, K+T, image_h, image_w,
                                              3), dtype="float32")
                        diff_batch = np.zeros((batch_size, K-1, image_h, image_w,
                                               3), dtype="float32")
                        accel_batch = np.zeros((batch_size, K-2, image_h, image_w,
                                                3), dtype="float32")
                        
                        tdirs = np.array(list(data_dict.keys()))[batchidx]
                        output = parallel(delayed(load_kitti_data)(d, data_dict[d],(image_h, image_w), K, T) for d in tdirs)
                        
                            seq_batch[i] = output[i][0]
                            diff_batch[i] = output[i][1]
                            accel_batch[i] = output[i][2]
                        
                       

                        if model_name == 'VNET':
                          model_input = {model.velocity: diff_batch,
                                              model.xt: seq_batch[:, K-1, :, :],
                                              model.target: seq_batch}
                        elif model_name == 'VANET' or model_name == 'VANET_ntd':
                          model_input = {model.velocity: diff_batch,
                                                model.accelaration: accel_batch,
                                                model.xt: seq_batch[:, K-1, :, :],
                                                model.target: seq_batch}
                        
                        if train_gen_only:
                            _, summary_str = sess.run([g_optim, g_sum],
                                                        feed_dict= model_input)
                            writer.add_summary(summary_str, counter)
                            
                            errG_L_sum = model.L_p.eval(model_input)
                            
                            errG_L_stgdl_sum = model.L_stgdl.eval(model_input)

                            print(
                                "Iters: [%2d] , Error_L_p: %.8f, Error_L_stgdl: %.8f"
                                % (iters, errG_L_sum, errG_L_stgdl_sum)
                                )
                        else:
                            
                            if updateD:
                                _, summary_str = sess.run([d_optim, d_sum],
                                                            feed_dict=model_input)
                            writer.add_summary(summary_str, counter)
                            
                            if updateG:
                                
                                _, summary_str = sess.run([g_optim, g_sum],
                                                            feed_dict=model_input)
                            writer.add_summary(summary_str, counter)
                            errG_L_sum = model.L_p.eval(model_input)
                            
                            errG_L_stgdl_sum = model.L_stgdl.eval(model_input)


                            errD_fake = model.d_loss_fake.eval(model_input)
                            errD_real = model.d_loss_real.eval(model_input)
                            errG = model.L_gen.eval(model_input)

                            if errD_fake < margin or errD_real < margin:
                                updateD = False
                                print("Not! updating Discriminator")
                            if errD_fake > (1.-margin) or errD_real > (1.-margin):
                                updateG = False
                                print("Not! updating generator")
                            if not updateD and not updateG:
                                updateD = True
                                updateG = True
                                print("Updating both Generator and Discriminator")
                            print("Updating Generator: "+str(updateG))
                            print("Updating Discriminator: "+str(updateD))
                            print(
                                    "Iters: [%2d], d_loss: %.8f, L_GAN: %.8f, errD_fake: %.8f, errD_real: %.8f" 
                                    % (iters, errD_fake+errD_real,errG, errD_fake,errD_real)
                                )
                            print(
                                    "Iters: [%2d], reconstruction_loss: %.8f, Error_L_p: %.8f, Error_L_stgdl: %.8f" 
                                    % (iters, errG_L_sum+errG_L_stgdl_sum,errG_L_sum, errG_L_stgdl_sum)
                                )

                        
                        counter+=1

                        if np.mod(counter, 100) == 1:
                            samples = sess.run([model.G],
                                               feed_dict=model_input)[0]
                            samples = samples[0]
                            sbatch = seq_batch[0, K:, :, :]
                            samples = np.concatenate((samples, sbatch), axis=0)
                            print("Saving sample ...")
                            save_images(samples[:, :, :, ::-1], [2, T],
                                        samples_dir+"train_%s.png" % (iters))
                            if len(rem_gen_samples) > 1000:
                                f = rem_gen_samples.pop(0)
                                if os.path.exists(f):
                                    os.remove(f)
                            rem_gen_samples.append(os.path.join(samples_dir, "train_%s.png" % (iters)))
                            #rem_gen_samples.append(samples_dir+"train_%s.png" % (iters))
                        if np.mod(counter, 500) == 0:
                            model.save(sess, checkpoint_dir, counter)

                        iters += 1
                print("Epoch time: [%4.4f]"% (time.time() - t0))

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
    parser.add_argument("--K", type=int, dest="K",
                        default=5, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=150000, help="Number of iterations")
    parser.add_argument("--gpu", type=int,  dest="gpu", required=False,
                        default=0, help="GPU device id")
    parser.add_argument("--beta1", type=float,  dest="beta1", required=False,
                        default=0.9, help="beta1 decay rate")
    parser.add_argument("--train_gen_only", default=False, action='store_true')
    parser.add_argument("--iters_start", type=int,  dest="iters_start", required=False, default=0, help='iteration_starts')

    args = parser.parse_args()
    main(**vars(args))
