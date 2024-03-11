
import os
import glob
import shutil
import numpy as np

max_seq_len = 8

old_dir = "/Users/daithyren/Desktop/School/graduate/UW/2024_winter/CSE455_computer_vision/final_proj/datasets/hmdb51_org_256x256q5"
new_dir = "/Users/daithyren/Desktop/School/graduate/UW/2024_winter/CSE455_computer_vision/final_proj/datasets/hmdb51_org_256x256q5_l{}".format(max_seq_len)

# 新的实现考虑到old_dir下的train, test, val文件夹
data_types = ['train', 'test', 'val']

os.mkdir(new_dir)

src_jpgs = 0
tgt_jpgs = 0

for data_type in data_types:
    
    data_type_dir = os.path.join(old_dir, data_type)
    classes = [j for j in glob.glob(os.path.join(data_type_dir, "*")) if os.path.isdir(j)]

    for class_folder in classes:
        c = os.path.split(class_folder)[-1]
        new_c = os.path.join(new_dir, c)

        if not os.path.exists(new_c):
            os.mkdir(new_c)

        for video_folder in glob.glob(os.path.join(class_folder, "*")):
            v = os.path.split(video_folder)[-1]
            new_v = os.path.join(new_c, v)

            print("ss")
            jpgs = [j for j in glob.glob(os.path.join(video_folder, "*.jpg"))]

            n_jpgs = len(jpgs)
            src_jpgs += n_jpgs

            if n_jpgs <= max_seq_len:
                tgt_jpgs += n_jpgs
                shutil.copytree(video_folder, new_v)
            else:
                os.mkdir(new_v)
                jpgs.sort()

                idx_f = np.linspace(0, n_jpgs-1, num=max_seq_len)
                idxs = [int(f) for f in idx_f]
                # print(len(idxs))
                for i in range(len(idxs)):
                    src = jpgs[idxs[i]]
                    tgt = os.path.join(new_v, "{:08d}.jpg".format(i+1))
                    shutil.copy(src, tgt)
                    tgt_jpgs += 1

print("Reduced {} to {} jpgs".format(src_jpgs, tgt_jpgs))
