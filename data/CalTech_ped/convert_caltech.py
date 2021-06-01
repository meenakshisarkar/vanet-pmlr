import os
import sys
from os import listdir, makedirs, system
from os.path import exists
image_size = 64 
data_path = os.getcwd()+'/CalTech_ped/'
# classes = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'}

for d1 in os.listdir(data_path):
    # dir2= (os.path.join(data_path, d1))
    print(d1) 
    for d2 in os.listdir(os.path.join(data_path, d1)):   
            data_dir= (os.path.join(data_path,d1, d2))
            filename= d2.split(".")
            savedir=data_path+"/processed/test/"+d1+'/'+filename[0]
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            cmd = ("ffmpeg -i "+data_dir+" -r 30 -f image2 -s 640x480  "+savedir+
                            "/img_%010d.png")
            system(cmd)