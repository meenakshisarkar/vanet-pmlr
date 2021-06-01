# VANET
# Abstract
 We propose a novel deep learning framework that focuses on decomposing the motion or the flow of the pixels from the background for an improved and longer prediction of video sequences. We propose to generate multi-timestep pixel level prediction using a framework that is trained to learn the temporal and spatial dependencies encoded in  video data separately. The proposed framework called Velocity Acceleration Network or VANet is  capable of predicting long term video frames for the static scenario, where the camera is stationary, as well as the dynamic partially observable cases, where the camera is mounted on a moving platform (cars or robots). This framework decomposes the flow of the image sequences into velocity and acceleration maps and learns the temporal transformations using a convolutional LSTM network. Our detailed empirical study on three different  datasets (BAIR, KTH and KITTI) shows that conditioning recurrent networks like LSTMs with higher order optical flow maps results in improved inference capabilities for videos. 

# Citation
If you are using VANet, please cite our proposal paper as:
@inproceedings{vanet,
  title={Decomposing camera and object motion for an improved video sequence prediction},
  author={Meenakshi Sarkar and Debasish Ghose},
  booktitle={Pre-registration workshop NeurIPS (2020), Vancouver, Canada},
  pages={},
  year={2020}
}
