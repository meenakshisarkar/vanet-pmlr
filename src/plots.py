import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
# model_spec="KITTI_Full_VANET_GPU_id=1_image_w=208_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration=150000_beta10.9_wo_campus"
# model_spec="BAIR_Full_VANET_GPU_id=1_image_h=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration150000"
# model_spec='KTH_VANET_GPU_id=1_image_w=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.001_lr=0.0001_no_iteration=150000_beta10.9'

# model_spec="KITTI_Full_VANET_GPU_id=0_image_w=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration=150000_beta10.9_wo_campus"
def main(lr, batch_size, alpha, beta, image_h, image_w, vid_type, K,
	     T, num_iter, gpu,  model_name,beta1,dataset,train_timesteps):
	# prefix = (dataset+"_{}".format(model_name)
	#           + "_GPU_id="+str(gpu)
	#           + "_image_w="+str(image_w)
	#           + "_K="+str(K)
	#           +  "_FutureT="+str(T)
	#           + "_batch_size="+str(batch_size)
	#           + "_alpha="+str(alpha)
	#           + "_beta="+str(beta)
	#           + "_lr="+str(lr)
	#           +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1))
	
	# print("\n"+prefix+"\n")
	if dataset.split('_')[0]== 'KITTI':
		model_name=["VANET","VNET"]
		color=['-r','g']
		model_number=["model-150000","model-150000"]
		beta1_lst=[0.9,0.9]
		beta_lst=[0.0001,0.0001]
		gpu_lst=[1,1]
		K_list= [5,5]
		batch_size_lst=[8,8]
		# model_name=["VANET","VANET_ntd","VNET"]
		# color=['-r','-b', 'g']
		# model_number=["model-150000","model-150000","model-150000"]
		# beta1_lst=[0.9,0.9, 0.9]
		# beta_lst=[0.0001,0.0001,0.0001]
		# gpu_lst=[1,1,1]
		# K_list= [5,5,5]
		# batch_size_lst=[8,8,8]
		T=25
	elif dataset.split('_')[0]=='KTH':
		model_name=["VANET","VANET_ntd","VNET"]
		color=['-r','-b', 'g']
		model_number=["model-150000","model-150000","model-150000"]
		beta1_lst=[0.9,0.9, 0.5]
		beta_lst=[0.001,0.001,0.02]
		gpu_lst=[0,0,1]
		K_list= [10,10,10]
		batch_size_lst=[8,8,8]
		T=20
	elif dataset.split('_')[0]=='BAIR':
		model_name=["VANET"]
		color = ['-r']
		model_number=["model-250000"]
		beta1_lst=[ 0.5]
		beta_lst=[0.0001]
		gpu_lst=[0]
		K_list= [10]
		batch_size_lst=[8,8,8]
		T=10
	time=range(0,10)
	# model_name=["VANET","VNET"]
	# color=['-r','-b']
	# model_number=["model-200000", "model-200000"]
	# beta1_lst=[0.5,0.5]
	# beta_lst=[0.0001,0.0001]
	# gpu_lst=[0,0]
	# K_list= [10,10]
	# batch_size_lst=[8,8]
	# model_number1="model-150000"
	# model_number2="model-149500"
	# model_number3="model-148500"
	fig, axis= plt.subplots(1,3)
	for model, c, beta, beta1, gpu, K, batch_size, model_no in zip(model_name, color, beta_lst,beta1_lst,gpu_lst, K_list,batch_size_lst, model_number):
		# for model_no in model_number:
			prefix = (dataset+"_{}".format(model)
	          + "_GPU_id="+str(gpu)
	          + "_image_w="+str(image_w)
	          + "_K="+str(K)
	          +  "_FutureT="+str(T)
	          + "_batch_size="+str(batch_size)
	          + "_alpha="+str(alpha)
	          + "_beta="+str(beta)
	          + "_lr="+str(lr)
	          +"_no_iteration="+str(num_iter)+"_beta1"+str(beta1))
			data=np.load('../results/quantitative/'+dataset.split('_')[0]+'/'+prefix+'/'+model_no+'/results_model='+model+'.npz')
			ssim= np.mean(data['ssim'], axis=0)
			psnr= np.mean(data['psnr'],axis=0)
			vgg16_csim= np.mean(data['vgg16_csim'], axis=0)
			fvd_score=data['fvd_score']
			print(fvd_score)
			# fig1, ax1= plt.subplot()
			axis[0].plot(time,ssim[:20], c)
			# plt.hold(True)
			# plt.plot(time,ssim3, '-g')
			axis[1].plot(time,psnr[:20], c)
			axis[2].plot(time,vgg16_csim[:20], c)

			# plt.hold(True)

	plt.show()

	# plt.hold(False)

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
	                    default=10, help="Number of steps to observe from the past")
	parser.add_argument("--T", type=int, dest="T",
	                    default=20, help="Number of steps into the future")
	parser.add_argument("--num_iter", type=int, dest="num_iter",
	                    default=250000, help="Number of iterations")
	parser.add_argument("--gpu", type=int,  dest="gpu", required=False,
	                    default=0, help="GPU device id")
	parser.add_argument("--beta1", type=float,  dest="beta1", required=False,
	                    default=0.9, help="beta1 decay rate")
	parser.add_argument("--train_timesteps", type=int,  dest="train_timesteps", required=False,
	                      default=10, help="future time steps")
	parser.add_argument("--dataset", type=str,  dest="dataset", required=False,
	                      default='KTH', help="Specify the name of the dataset")
	args = parser.parse_args()
	main(**vars(args))



# model_name1='VANET'
# model_name2='VNET'
# model_spec1="KITTI_Full_"+model_name1+"_GPU_id=0_image_w=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration=150000_beta10.9_wo_campus"
# model_spec2="KITTI_Full_"+model_name2+"_GPU_id=0_image_w=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration=150000_beta10.9_wo_campus"
# time=range(0,25)
# data1=np.load('../results/quantitative/KITTI/'+model_spec1+'/'+model_number1+'/results_model='+model_name1+'.npz')
# data2=np.load('../results/quantitative/KITTI/'+model_spec2+'/'+model_number1+'/results_model='+model_name2+'.npz')
# # data3=np.load('../results/quantitative/KITTI/'+model_spec+'/'+model_number3+'/results_model='+model_name+'.npz')
# # data=np.load('../results/quantitative/BAIR/'+model_spec+'/'+model_number+'/results_model=VANET.npz_FILES/psnr.npy')
# # data=np.load('../results/quantitative/BAIR/'+model_spec+'/'+model_number+'/psnr.npy')

# # data=np.load('../results/quantitative/KTH/'+model_spec+'/'+model_number+'/results_model=VANET.npz_FILES/ssim.npy')


# # data=np.load('../results/quantitative/KITTI/'+model_spec+'/psnr.npy')
# ssim1= np.mean(data1['ssim'], axis=0)
# psnr1= np.mean(data1['psnr'],axis=0)
# ssim2= np.mean(data2['ssim'], axis=0)
# psnr2= np.mean(data2['psnr'],axis=0)
# # ssim3= np.mean(data3['ssim'], axis=0)
# # psnr3= np.mean(data3['psnr'],axis=0)

# plt1= plt.figure(1)
# plt.plot(time,ssim1, '-r')
# plt.plot(time,ssim2,'-b')
# # plt.plot(time,ssim3, '-g')
# plt2 =  plt.figure(2)
# plt.plot(time,psnr1, '-r')
# plt.plot(time,psnr2, '-b')
# # plt.plot(time,psnr3, '-g')
# # plt.plot(data[117])

# plt.show()




