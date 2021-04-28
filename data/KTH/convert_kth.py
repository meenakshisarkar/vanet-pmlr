import os
import sys
from os import listdir, makedirs, system
from os.path import exists
image_size = 64 
data_path = "pwd"
# classes = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'}
for d1 in os.listdir(data_path): 
    if d1.split('.')[-1]=='avi' :    
        data_dir= (os.path.join(data_path, d1))
        d2= d1.split("_")
        # print(d2[0].split("n")[1])
        # exit()
        if int(d2[0].split("n")[1])<17:
          savedir=data_path+"/processed/train/"+d2[1]+'/'+d2[0]
        else:
          savedir=data_path+"/processed/test/"+d2[1]+'/'+d2[0]
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        cmd = ("ffmpeg -i "+data_dir+" -r 25 -f image2 -s 64x64  "+savedir+
                        "/img_%06d.png")
        system(cmd)


