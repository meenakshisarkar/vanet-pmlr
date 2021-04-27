import os
import sys
from os import listdir, makedirs, system
from os.path import exists
# opt = lapp[[  
#   --imageSize        (default 128)                size of image
#   --dataRoot         (default '/path/to/data/')  data root directory
# ]] 
# data_dir=[]
image_size = 64 
data_path = "KTH"
# classes = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'}
for d1 in os.listdir(data_path):      
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


# classes = {'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'}

# frame_rate = 25

# for _, class in pairs(classes) do
#   print(' ---- ')
#   print(class)

#   for vid in paths.iterfiles(data_root .. '/raw/' .. class) do
#     print(vid)
#     local fname = vid:sub(1,-12)
#     os.execute(('mkdir -p %s/processed/%s/%s'):format(data_root, class, fname))
#     os.execute( ('~/tools/ffmpeg-git-20170417-32bit-static/ffmpeg -i %s/raw/%s/%s -r %d -f image2 -s %dx%d  %s/processed/%s/%s/image-%%03d_%dx%d.png'):format(data_root, class, vid, frame_rate, image_size, image_size, data_root, class, fname, image_size, image_size))
#   end
# end
