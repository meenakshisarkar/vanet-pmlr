import os
import tensorflow as tf

import numpy as np
from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *
import tensorflow.contrib.slim as slim


class VANET_v2(object):
    def __init__(self, image_size=[128, 128], batch_size=32, c_dim=3, timesteps=10, F=10,
                 checkpoint_dir=None, training=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.timesteps = timesteps
        self.F = F
        self.checkpoint_dir = checkpoint_dir
        self.is_train = training
        self.filters= 32
        self.df_dim=32

        self.vel_shape = [batch_size, timesteps - 1, image_size[0], image_size[1], c_dim]
        self.acc_shape = [batch_size, timesteps - 2, image_size[0], image_size[1], c_dim]
        self.xt_shape = [batch_size, image_size[0], image_size[1], c_dim]
        self.gt_shape = [batch_size, timesteps, image_size[0], image_size[1], c_dim]
        self.target_shape = [batch_size, self.timesteps+self.F, image_size[0], image_size[1], c_dim]

        self.create_model()

    def create_model(self):
        self.velocity = tf.placeholder(tf.float32, self.vel_shape, name='velocity')
        self.accelaration = tf.placeholder(tf.float32, self.acc_shape, name='accelaration')
        self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        vel_LSTM = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                     [3, 3], self.filters * 4)
        acc_LSTM = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                          [3, 3], self.filters * 4)
        predict = self.forward_model(self.velocity, self.accelaration, self.xt, vel_LSTM, acc_LSTM)
        self.G = tf.concat(axis=1, values=predict)

        dis_reuse= False
        if self.is_train:
            _D_real= []
            _D_logits_real=[]
            _D_fake =[]
            _D_logits_fake =[]
            _Dis_in_img=inverse_transform(tf.reshape(tf.transpose(self.target[:,:self.timesteps,:,:,:], [0,2,3,1,4]),
                                            [self.batch_size,self.image_size[0], self.image_size[1],-1]))
            _Dis_target_img=inverse_transform(tf.reshape(tf.transpose(self.target[:,self.timesteps:self.timesteps+self.F,:,:,:],[0,2,3,4,1]),
                                            [self.batch_size,self.image_size[0], self.image_size[1],-1]))
            _Dis_gen_img=inverse_transform(tf.reshape(tf.transpose(self.G, [0,2,3,1,4]),
                                            [self.batch_size,self.image_size[0], self.image_size[1],-1]))
            _Dis_real_img= tf.concat([_Dis_in_img,_Dis_target_img],axis=3)
            _Dis_fake_img= tf.concat([_Dis_in_img,_Dis_gen_img],axis=3)
            # self.D_real_, self.D_logits_real_= self.discriminator(_Dis_real_img, reuse= dis_reuse) 
            # print self.D_real_.shape
            # dis_reuse= True
            # self.D_fake_, self.D_logits_fake_ = self.discriminator(_Dis_fake_img, reuse= dis_reuse)

            # for l in xrange(self.F):
            #     in_img=tf.reshape(tf.transpose(self.target[:,self.timesteps+l-2:self.timesteps+l,:,:,:], [0,2,3,1,4]),
            #                                 [self.batch_size,self.image_size[0], self.image_size[1],-1])
            #     target_img=tf.reshape(self.target[:,self.timesteps+l,:,:,:],
            #                                 [self.batch_size,self.image_size[0], self.image_size[1],-1])
            #     # target_img=tf.reshape(tf.transpose(self.target[:,self.timesteps+l,:,:,:], [0,2,3,1,4]),
            #     #                             [self.batch_size,self.image_size[0], self.image_size[1],-1])
            #     print self.G.shape
            #     gen_img=tf.reshape(self.G[:,l,:,:,:],
            #                                 [self.batch_size,self.image_size[0], self.image_size[1],-1])
            #     # gen_img=tf.reshape(tf.transpose(self.G[:,l,:,:,:], [0,2,3,1,4]),
            #     #                             [self.batch_size,self.image_size[0], self.image_size[1],-1])
            #     real_img= tf.concat([in_img,target_img],axis=3)
            #     fake_img= tf.concat([in_img,gen_img],axis=3)

            #     ########Rethink the variable scope part
            #     self.D_real_, self.D_logits_real_= self.discriminator(real_img, reuse= dis_reuse)
            #     if l==0: 
            #         dis_reuse= True
            #     self.D_fake_, self.D_logits_fake_ = self.discriminator(fake_img, reuse= dis_reuse)
            #     _D_real.append(self.D_real_)
            #     _D_logits_real.append(self.D_logits_real_)
            #     _D_fake.append(self.D_fake_)
            #     _D_logits_fake.append(self.D_logits_fake_)
            

            #################reconstruction losses
            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target[:, self.timesteps:, :, :, :]))
            self.L_stgdl= stgdl(self.G, self.target[:,self.timesteps:, :, :,  :],1.0, self.image_size[0],channel_no=1)

            self.reconst_loss= self.L_p+self.L_stgdl


            ################# Generative and adversarial losses
            # self.d_loss_real = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=self.D_logits_real, labels=tf.ones_like(self.D_real))) 
            # self.d_loss_fake = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=self.D_logits_fake, labels=tf.zeros_like(self.D_fake)))
            # self.d_loss= self.d_loss_real+self.d_loss_fake

            # self.L_gen= tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=self.D_logits_fake, labels=tf.ones_like(self.D_fake)))

            ################## Loss summery
            self.L_sum = tf.summary.scalar("reconst_loss", self.reconst_loss)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_stgdl_sum = tf.summary.scalar("L_stgdl", self.L_stgdl)
            # self.L_Gen_sum = tf.summary.scalar("L_gen", self.L_gen)
            # self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            # self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            # self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(self.t_vars, print_info=True)
            self.g_vars = [var for var in self.t_vars if 'Dis' not in var.name]
            # self.d_vars = [var for var in self.t_vars if 'Dis' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()))
            print("Number of parameters: %d" % num_param)
        self.saver = tf.train.Saver(max_to_keep=10)



    def forward_model(self, vel_in, acc_in, xt, vel_LSTM, acc_LSTM):
        vel_state = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 256])  #this takes double the no of channels of the state as it concatinates the c and h states of ConvLSTM
        acc_state = tf.zeros([self.batch_size, self.image_size[0] / 8, self.image_size[1] / 8, 256])
        reuse_vel = False
        reuse_acc= False

        # Encoder
        for t in xrange(self.timesteps - 1):
            h_vel_out, vel_state, vel_res_in = self.vel_enc(vel_in[:, t, :, :, :], vel_state, vel_LSTM, name= "vel_enc", reuse=reuse_vel)
            if t<self.timesteps-2:
                h_acc_out, acc_state, acc_res_in = self.acc_enc(acc_in[:, t, :, :, :], acc_state, acc_LSTM, name= "acc_enc", reuse=reuse_acc)
            reuse_vel = True
            reuse_acc = True
        
        predict = []
        for t in xrange(self.F):
            if t==0:
                h_con_state, con_res_in= self.content_enc(xt, name="content_enc", reuse=False)
                cont_conv = self.conv_layer(h_con_state, h_acc_out, h_vel_out, name="conv_layer", reuse= False)
                res_conv = self.res_conv_layer(con_res_in, acc_res_in, vel_res_in,name="res_conv_layer", reuse= False)
                x_tilda= self.dec_layer(cont_conv,res_conv,name="dec_layer", reuse = False)
                print("I crossed decoding for t==0 layer")
                print vel_in.shape
                vel_in = vel_in[:,self.timesteps-2,:,:,:]
                acc_in = acc_in[:,self.timesteps-3,:,:,:]
                # vel_in = tf.squeeze(vel_in[:,self.timesteps-2,:,:,:], [1])
                # acc_in = tf.squeeze(acc_in[:,self.timesteps-3,:,:,:], [1])
            else:
                print("I started t==1")
                h_vel_out, vel_state, vel_res_in = self.vel_enc(vel_in, vel_state, vel_LSTM,name= "vel_enc", reuse=reuse_vel)
                h_acc_out, acc_state, acc_res_in = self.acc_enc(acc_in, acc_state, acc_LSTM, name= "acc_enc", reuse=reuse_acc)
                h_con_state, con_res_in= self.content_enc(xt, name="content_enc", reuse=True)
                cont_conv = self.conv_layer(h_con_state, h_acc_out, h_vel_out, name="conv_layer",reuse= True)
                res_conv= self.res_conv_layer(con_res_in, acc_res_in, vel_res_in, name="res_conv_layer", reuse= True)
                x_tilda= self.dec_layer(cont_conv,res_conv, name="dec_layer", reuse= True)
            vel_in_past= vel_in
            vel_in= x_tilda- xt
            acc_in= vel_in - vel_in_past
            xt=x_tilda
            predict.append(tf.reshape(x_tilda,[self.batch_size,1, self.image_size[0], self.image_size[1], self.c_dim]))


        return predict
                

    def vel_enc(self, vel_in, vel_state, vel_LSTM, name, reuse):
        with tf.variable_scope(name, reuse):
            vel_res_in = []
            conv1 = relu(conv2d(vel_in, output_dim=self.filters, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='vel_conv1', reuse=reuse))
            vel_res_in.append(conv1)
            pool1 = MaxPooling(conv1, [2, 2], stride=2)

            conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='vel_conv2', reuse=reuse))
            vel_res_in.append(conv2)
            pool2 = MaxPooling(conv2, [2, 2], stride=2)

            conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=3, k_w=3,
                                d_h=1, d_w=1, name='vel_conv3', reuse=reuse))
            vel_res_in.append(conv3)
            pool3 = MaxPooling(conv3, [2, 2],stride=2)
            print(pool3.shape)
            h1_state, vel_state = vel_LSTM(pool3, vel_state, scope='vel_lstm1', reuse=reuse)
            h2_state, vel_state = vel_LSTM(h1_state, vel_state, scope='vel_lstm2', reuse=reuse)
            h_vel_out, vel_state = vel_LSTM(h2_state, vel_state, scope='vel_lstm3', reuse=reuse)
        return h_vel_out, vel_state, vel_res_in                  #h_vel_out

    def acc_enc(self, acc_in, acc_state, acc_LSTM, name, reuse):
        with tf.variable_scope(name, reuse):
            acc_res_in = []
            conv1 = relu(conv2d(acc_in, output_dim=self.filters, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='acc_conv1', reuse=reuse))
            acc_res_in.append(conv1)
            pool1 = MaxPooling(conv1, [2, 2],stride=2)

            conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='acc_conv2', reuse=reuse))
            acc_res_in.append(conv2)
            pool2 = MaxPooling(conv2, [2, 2],stride=2)

            conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=3, k_w=3,
                                d_h=1, d_w=1, name='acc_conv3', reuse=reuse))
            acc_res_in.append(conv3)
            pool3 = MaxPooling(conv3, [2, 2],stride=2)
            h1_state, acc_state = acc_LSTM(pool3, acc_state, scope='acc_lstm1', reuse=reuse)
            h2_state, acc_state = acc_LSTM(h1_state, acc_state, scope='acc_lstm2', reuse=reuse)
            h_acc_out, acc_state = acc_LSTM(h2_state, acc_state, scope='acc_lstm3', reuse=reuse)
        return h_acc_out, acc_state, acc_res_in

    def content_enc(self,xt,name,reuse):
        with tf.variable_scope(name, reuse):
            con_res_in = []
            conv1 = relu(conv2d(xt, output_dim=self.filters, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='con_conv1', reuse=reuse))
            con_res_in.append(conv1)               ### 128*128*32
            pool1 = MaxPooling(conv1, [2, 2],stride=2)      ####64*64*32

            conv2 = relu(conv2d(pool1, output_dim=self.filters * 2, k_h=5, k_w=5,
                                d_h=1, d_w=1, name='con_conv2', reuse=reuse))
            con_res_in.append(conv2)               ### 64*64*64
            pool2 = MaxPooling(conv2, [2, 2],stride=2)      ### 32*32*64

            conv3 = relu(conv2d(pool2, output_dim=self.filters * 4, k_h=3, k_w=3,
                                d_h=1, d_w=1, name='con_conv3', reuse=reuse))
            con_res_in.append(conv3)              #### 32*32*128
            pool3 = MaxPooling(conv3, [2, 2],stride=2)     #### 16*16*128
        return pool3, con_res_in

    def conv_layer(self,h_con_state, h_acc_out, h_vel_out,name, reuse):
        # print h_con_state.shape, h_acc_out.shape
        with tf.variable_scope(name, reuse):
            "$Need to be updated$"
            motion_in= tf.concat([h_vel_out, h_acc_out], axis= 3 )
            motion_filter= relu(conv2d(motion_in, output_dim= h_acc_out.shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "conv_layer1", reuse=reuse))
            cont_conv1=relu(conv2d(tf.concat([h_con_state,motion_filter],axis=3),output_dim= h_acc_out.shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "conv_layer2", reuse=reuse ))
            cont_conv2=relu(conv2d(cont_conv1,output_dim= h_acc_out.shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "conv_layer3", reuse=reuse ))
        return cont_conv2

    def res_conv_layer(self, con_res_in, acc_res_in, vel_res_in,name, reuse):
        with tf.variable_scope(name, reuse):
            "$Need to be updated$"
            res_conv_out=[]
            #no_layers= len(con_res_in)
            res_motion_in1= tf.concat([vel_res_in[0], acc_res_in[0]], axis= 3 )
            res_motion_filter1= relu(conv2d(res_motion_in1, output_dim= acc_res_in[0].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer1_1", reuse=reuse))
            res_cont_conv1_1=relu(conv2d(tf.concat([con_res_in[0],res_motion_filter1],axis=3),output_dim= acc_res_in[0].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer1_2", reuse=reuse ))
            res_cont_conv1_2=relu(conv2d(res_cont_conv1_1,output_dim= acc_res_in[0].shape[-1], k_h=5, k_w=5,
                                            d_h=1, d_w=1, name= "res_conv_layer1_3", reuse=reuse ))
            res_conv_out.append(res_cont_conv1_2)

            res_motion_in2= tf.concat([vel_res_in[1], acc_res_in[1]], axis= 3 )
            res_motion_filter2= relu(conv2d(res_motion_in2, output_dim= acc_res_in[1].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer2_1", reuse=reuse))
            res_cont_conv2_1=relu(conv2d(tf.concat([con_res_in[1],res_motion_filter2],axis=3),output_dim= acc_res_in[1].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer2_2", reuse=reuse ))
            res_cont_conv2_2=relu(conv2d(res_cont_conv2_1,output_dim= acc_res_in[1].shape[-1], k_h=5, k_w=5,
                                            d_h=1, d_w=1, name= "res_conv_layer2_3", reuse=reuse ))
            res_conv_out.append(res_cont_conv2_2)

            res_motion_in3= tf.concat([vel_res_in[2], acc_res_in[2]], axis= 3 )
            res_motion_filter3= relu(conv2d(res_motion_in3, output_dim= acc_res_in[2].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer3_1", reuse=reuse))
            res_cont_conv3_1=relu(conv2d(tf.concat([con_res_in[2],res_motion_filter3],axis=3),output_dim= acc_res_in[2].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer3_2", reuse=reuse ))
            res_cont_conv3_2=relu(conv2d(res_cont_conv3_1,output_dim= acc_res_in[2].shape[-1], k_h=3, k_w=3,
                                            d_h=1, d_w=1, name= "res_conv_layer3_3", reuse=reuse ))
            res_conv_out.append(res_cont_conv3_2)
            # res_conv1_1= convOp_mod(con_res_in[0], acc_res_in[0] ,reuse,name= 'res_conv1_1')   #ConvOp makes the computing extremely slow. Need a better way to evaluate cross conv
            # res_conv1_2= convOp_mod(res_conv1_1, vel_res_in[0], reuse,name='res_conv1_2')
            # res_conv_out.append(res_conv1_2)
            # res_conv2_1= convOp_mod(con_res_in[1], acc_res_in[1] ,reuse,name= 'res_conv2_1')   #ConvOp makes the computing extremely slow. Need a better way to evaluate cross conv
            # res_conv2_2= convOp_mod(res_conv2_1, vel_res_in[1], reuse,name='res_conv2_2')
            # res_conv_out.append(res_conv2_2)
            # res_conv3_1= convOp_mod(con_res_in[2], acc_res_in[2] ,reuse,name= 'res_conv3_1')   #ConvOp makes the computing extremely slow. Need a better way to evaluate cross conv
            # res_conv3_2= convOp_mod(res_conv3_1, vel_res_in[2], reuse,name='res_conv3_2')
            # res_conv_out.append(res_conv3_2)
        return res_conv_out
            
    def dec_layer(self, cont_conv,res_conv,name, reuse):
        with tf.variable_scope(name, reuse):

            shape1 = [self.batch_size, self.image_size[0]/4,
                                            self.image_size[1]/4, self.filters*4]
            up_samp1 = FixedUnPooling(cont_conv, [2, 2])
            decode1_1= relu(deconv2d(up_samp1,
                                        output_shape=shape1, k_h=3, k_w=3,
                                        d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
            #### 128 channels, image 32*32
            shape2 = [self.batch_size, self.image_size[0]/2,
                                            self.image_size[1]/2, self.filters*2]
            decod2_1 = tf.concat(axis=3, values=[decode1_1, res_conv[2]])
            up_samp2 = FixedUnPooling(decod2_1, [2, 2])
            decode2_2 = relu(deconv2d(up_samp2,
                                    output_shape=shape2, k_h=5, k_w=5,
                                    d_h=1, d_w=1, name='dec_deconv2_2', reuse=reuse))
            ### 64 channels image 64 *64
            shape3 = [self.batch_size, self.image_size[0],
                                            self.image_size[1], self.filters]
            decod3_1 = tf.concat(axis=3, values=[decode2_2, res_conv[1]])
            up_samp3 = FixedUnPooling(decod3_1, [2, 2])
            decode3_2 = relu(deconv2d(up_samp3,
                                    output_shape=shape3, k_h=5, k_w=5,
                                    d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse))
            #### 32 channels image 128*128
            decod4_1 = tf.concat(axis=3, values=[decode3_2, res_conv[0]])
            decode4_2 = relu(deconv2d(decod4_1,
                                    output_shape=shape3, k_h=5, k_w=5,
                                    d_h=1, d_w=1, name='dec_deconv4_2', reuse=reuse))

            decod5_1 = tf.concat(axis=3, values=[decode4_2, self.xt])
            decode_out = relu(conv2d(decod5_1, output_dim=self.c_dim, k_h=1, k_w=1,
                                d_h=1, d_w=1, name='decode_out', reuse=reuse))
        
        return decode_out

    def discriminator(self, image, name= 'Dis', reuse= False):
        print reuse
        with tf.variable_scope(name, reuse):
            h0 = lrelu(conv2d(image, output_dim=self.df_dim,k_h=5, k_w=5,
                                        d_h=1, d_w=1, name='dis_h0_conv', reuse=reuse))
            h0_pool = MaxPooling(h0, [2, 2],stride=2) 
            h1 = lrelu(batch_norm(conv2d(h0_pool, output_dim=self.df_dim*2, k_h=5, k_w=5,
                                        d_h=1, d_w=1, name='dis_h1_conv',reuse=reuse),reuse=reuse, name="bn1"))
            h1_pool = MaxPooling(h1, [2, 2],stride=2) 
            h2 = lrelu(batch_norm(conv2d(h1_pool, output_dim=self.df_dim*4,k_h=5, k_w=5,
                                        d_h=1, d_w=1, name='dis_h2_conv',reuse=reuse),reuse=reuse, name="bn2"))
            h2_pool = MaxPooling(h2, [2, 2],stride=2) 
            h3 = lrelu(batch_norm(conv2d(h2_pool, output_dim=self.df_dim*8,k_h=3, k_w=3,
                                        d_h=1, d_w=1, name='dis_h3_conv',reuse=reuse),reuse=reuse, name="bn3"))
            h3_pool = MaxPooling(h3, [2, 2],stride=2) 
            h = linear(tf.reshape(h3_pool, [self.batch_size, -1]), self.F, reuse=reuse, name='dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "VANET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)


    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print ckpt, ckpt.model_checkpoint_path
        if ckpt and ckpt.model_checkpoint_path:
            print "I am inside checkpoint"
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: "+str(model_name))
            return True, model_name
        else:
            return False, None
