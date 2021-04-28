import numpy as np
import matplotlib.pyplot as plt
# model_spec="best_KITTI_Full_VANET_GPU_id=1_image_w=208_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration=150000_beta10.9_wo_campus"
model_spec="BAIR_Full_VANET_GPU_id=1_image_h=64_K=10_T=10_batch_size=8_alpha=1.0_beta=0.0001_lr=0.0001_no_iteration150000"
model_number="model-150000"
# data=np.load('../results/quantitative/KITTI/'+model_spec+'/'+model_number+'/results_model=VANET.npz_FILES/psnr.npy')
data=np.load('../results/quantitative/BAIR/'+model_spec+'/'+model_number+'/results_model=VANET.npz_FILES/psnr.npy')
# data=np.load('../results/quantitative/BAIR/'+model_spec+'/'+model_number+'/psnr.npy')

# data=np.load('../results/quantitative/KITTI/'+model_spec+'/psnr.npy')
plt.plot(np.mean(data, axis=0))
# plt.plot(data[117])

plt.show()