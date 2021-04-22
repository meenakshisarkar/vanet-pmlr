import numpy as np
import matplotlib.pyplot as plt
model_spec="BAIR_Full_VANET_GPU_id=0_image_h=64_K=10_T=10_batch_size=16_alpha=1.0_beta=0.001_lr=0.0001_no_iteration150000"
data=np.load('../results/quantitative/BAIR/vanet_wo_gen_v1/'+model_spec+'/psnr.npy')
plt.plot(np.mean(data, axis=0))
# plt.plot(data[117])

plt.show()