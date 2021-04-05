import os
import argparse

def count_frames(base_dir):
    _, dirs, _ = next(os.walk(base_dir))
    frame_count = {}
    for d in dirs:
        fpath = os.path.join(base_dir, d, 'image_03/data')
        frame_count[d] = len([f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))])
    return frame_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, dest="base_dir",
        default='../data/KITTI/', help="path to kitti data")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    args = parser.parse_args()
    frame_count = count_frames(args.base_dir)
    with open('../data/KITTI/short_len_videos.txt', 'w') as f:
        for vid_file, count in frame_count.items():
            if count - args.K - args.T + 1 < 0:
                f.write('{} {}\n'.format(vid_file, count))
        
