import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio
import os

from vanet import VANET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def main(lr, batch_size, alpha, beta, image_h, vid_type, K,
         T, num_iter, gpu):
    data_path = "../data/KITTI/"
    with open(data_path+"train.txt", "r") as f:
        trainfiles = f.readlines()
    margin = 0.3
    updateD = True
    updateG = True
    iters = 1
    prefix = ("KITTI_Full_VANET"
              + "_image_h="+str(image_h)
              + "_K="+str(K)
              + "_T="+str(T)
              + "_batch_size="+str(batch_size)
              + "_alpha="+str(alpha)
              + "_beta="+str(beta)
              + "_lr="+str(lr)
              +"_no_iteration"+str(num_iter))

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

    if gpu == 0:
        gpus = False  # checking for GPU availability
    else:
        gpus = True

    image_w = int((image_h / 375.0) * 1242)

    # Selecting cpu or gpu "/gpu:%d"%gpu[0] if gpus else
    with tf.device("/gpu:0"):
        model = VANET(image_size=[image_h, image_w], c_dim=3,
                         timesteps=K, batch_size=batch_size, F=T, checkpoint_dir=checkpoint_dir)

        g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
            alpha*model.reconst_loss, var_list=model.g_vars)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    # (config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options if gpus else None))
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:                                     #if gpus else None

        tf.global_variables_initializer().run()

        success_load_model = model.load(sess, checkpoint_dir)
        print success_load_model[0]

        if success_load_model[0]:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge([model.L_p_sum,
                                  model.L_stgdl_sum, model.L_sum,])
        d_sum = tf.summary.merge([model.d_loss_real_sum,
                                  model.d_loss_fake_sum, model.d_loss_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters+1
        start_time = time.time()
        rem_ip_samples, rem_gen_samples, rem_model = [], [], []
        with Parallel(n_jobs=batch_size) as parallel:
            while iters < num_iter:
                mini_batches = get_minibatches_idx(
                    len(trainfiles), batch_size, shuffle=True)
                for _, batchidx in mini_batches:
                    if len(batchidx) == batch_size:
                        seq_batch = np.zeros((batch_size, K+T, image_h, image_w,
                                              1), dtype="float32")
                        diff_batch = np.zeros((batch_size, K-1, image_h, image_w,
                                               1), dtype="float32")
                        accel_batch = np.zeros((batch_size, K-2, image_h, image_w,
                                                1), dtype="float32")
                        t0 = time.time()
                        # Ts = np.repeat(np.array([T]), batch_size, axis=0)
                        # Ks = np.repeat(np.array([K]), batch_size, axis=0)
                        # paths = np.repeat(data_path, batch_size, axis=0)
                        tfiles = np.array(trainfiles)[batchidx]
                        # shapes = np.repeat(
                        #     np.array([image_ht]), batch_size, axis=0)
                        output = parallel(delayed(load_kitti_data)(f, data_path, image_h, K, T, vid_type)
                                          for f in tfiles)
                        print seq_batch[0].shape, output[0][0].shape
                        for i in xrange(batch_size):
                            seq_batch[i] = output[i][0]
                            diff_batch[i] = output[i][1]
                            accel_batch[i] = output[i][2]
                        print "I am at checkpoint batcave"
                        if iters%200==0:    
                            input_sample= seq_batch[0]
                            print("Saving input_sample ...")
                            save_images(input_sample[:K,:,:,::-1], [1, K],
                                            samples_dir+"image_inputs_to_network_mod%s.png" % (iters))
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
                            if len(rem_ip_samples) > 2:
                                for f in rem_ip_samples.pop(0):
                                    if os.path.exists(f):
                                        os.remove(f)
                            curr_sample = ["image_inputs_to_network_mod%s.png" % (iters),
                            "velo_inputs_to_network_mod%s.png" % (iters),
                            "accel_inputs_to_network_mod%s.png" % (iters)]
                            curr_sample = tuple([os.path.join(samples_dir, f) for f in curr_sample])
                            rem_ip_samples.append(curr_sample)

# need to change the input to the model and the indexing of the input images needs to be correct.model.target: seq_batch

                        if updateD:

                          _, summary_str = sess.run([d_optim, d_sum],
                                                     feed_dict={model.velocity: diff_batch,
                                                                model.accelaration: accel_batch,
                                                                model.xt: seq_batch[:,K-1,:,:,:],
                                                                model.target: seq_batch})
                          writer.add_summary(summary_str, counter)
                        print "here there"
                        if updateG:
                            _, summary_str = sess.run([g_optim, g_sum],
                                                      feed_dict={model.velocity: diff_batch,
                                                                 model.accelaration: accel_batch,
                                                                 model.xt: seq_batch[:, K-1, :, :],
                                                                 model.target: seq_batch})
                            writer.add_summary(summary_str, counter)
                        print "I am at checkpoint gotham"

                        print "ola"

                        errG_L_sum = model.L_p.eval({model.velocity: diff_batch,
                                                               model.accelaration: accel_batch,
                                                               model.xt: seq_batch[:,K-1,:,:,:],
                                                               model.target: seq_batch})
                        errG_L_stgdl_sum = model.L_stgdl.eval({model.velocity: diff_batch,
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
                            "Iters: [%2d] time: %4.4f, Error_L_p: %.8f, Error_L_stgdl: %.8f"
                            % (iters, time.time() - start_time, errG_L_sum, errG_L_stgdl_sum)
                        )

                        if (iters%100) == 0:
                            samples = sess.run([model.G],
                                               feed_dict={model.velocity: diff_batch,
                                                          model.accelaration: accel_batch,
                                                          model.xt: seq_batch[:, K-1, :, :, :],
                                                          model.target: seq_batch})[0]
                            samples = samples[0]
                            sbatch = seq_batch[0, K:, :, :]
                            samples = np.concatenate((samples, sbatch), axis=0)
                            print("Saving sample ...")
                            save_images(samples[:, :, :, ::-1], [2, T],
                                        samples_dir+"train_%s.png" % (iters))
                            if len(rem_gen_samples) > 2:
                                f = rem_gen_samples.pop(0)
                                if os.path.exists(f):
                                    os.path.remove(f)
                                rem_gen_samples.append(os.path.join(samples_dir, "train_%s.png" % (iters)))
                        if np.mod(iters, 500) == 0:
                            model.save(sess, checkpoint_dir, counter)

                        iters += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.02, help="GAN loss weight")
    parser.add_argument("--image_h", type=int, dest="image_h",
                        default=128, help="Frame height")
    parser.add_argument("--vid_type", type=str, dest="vid_type",
                        default='03', help="Grayscale/color, right/left stereo recordings")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=False,
                        default=0, help="GPU device id")

    args = parser.parse_args()
    main(**vars(args))
