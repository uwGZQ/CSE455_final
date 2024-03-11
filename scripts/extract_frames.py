import os
import subprocess
import json
from glob import glob
# from joblib import Parallel, delayed
import cv2

out_h = 256
out_w = 256
in_folder = '/Users/daithyren/Desktop/School/graduate/UW/2024_winter/CSE455_computer_vision/final_proj/datasets/hmdb51_org'
out_folder = '/Users/daithyren/Desktop/School/graduate/UW/2024_winter/CSE455_computer_vision/final_proj/datasets/hmdb51_org_{}x{}q5'.format(out_w,out_h)

split_dir = "/Users/daithyren/Desktop/School/graduate/UW/2024_winter/CSE455_computer_vision/final_proj/CSE455_final-main/splits/hmdb_ARN"

wc = os.path.join(split_dir, "*.txt")

# 确保输出目录存在
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

print("out_folder successful")

for fn in glob(wc):
    classes = []
    vids = []

    print(fn)
    if "train" in fn:
        cur_split = "train"
    elif "val" in fn:
        cur_split = "val"
    elif "test" in fn:
        cur_split = "test"

    with open(fn, "r") as f:
        data = f.readlines()
        c = [x.split('/')[-2].strip() for x in data]
        v = [x.split('/')[-1].strip().split('.')[0] for x in data]  # Assuming the file format is included
        vids.extend(v)
        classes.extend(c)

    split_out_folder = os.path.join(out_folder, cur_split)
    if not os.path.exists(split_out_folder):
        os.makedirs(split_out_folder)

    for c in set(classes):
        class_out_folder = os.path.join(split_out_folder, c)
        if not os.path.exists(class_out_folder):
            os.makedirs(class_out_folder)

    for v, c in zip(vids, classes):
        source_vid = os.path.join(in_folder, c, "{}.avi".format(v))
        extract_dir = os.path.join(out_folder, cur_split, c, v)

        if os.path.exists(extract_dir):
            continue
        else:
            os.makedirs(extract_dir)

        cap = cv2.VideoCapture(source_vid)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (out_w, out_h))
            frame_path = os.path.join(extract_dir, '{:08d}.jpg'.format(frame_count))
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()
