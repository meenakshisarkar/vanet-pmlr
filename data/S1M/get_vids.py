import os
import pytube


#where to save 
SAVE_PATH = "../../data/S1M/" #to_do 

#link of the video to be downloaded 
paths=open('train_partition.txt', 'r') #opening the file + ' --recode-video ' + ' mp4 '
lines = paths.readlines()
MAX_NUM_VIDS = 70000
for ln in range(MAX_NUM_VIDS):
  video_name = 'sports-1m_{0:09d}'.format(ln)
  link= lines[ln].split(' ')[0]
  os.system('youtube-dl '+ '-o ' +'temp_vid ' + link ) 
  os.system('ffmpeg -i  temp_vid.mkv -vf scale=360:240,setsar=1:1 '+ video_name+'.mp4' )
  os.system('rm temp_vid.mkv')
print('Task Completed!') 
