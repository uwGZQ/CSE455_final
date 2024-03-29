import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = [] # ground truth action labels
        self.videos = []    # list of lists of frame paths
        self.videos_flows_X = [] # list of lists of flow_x paths for optical flow
        self.videos_flows_Y = [] # list of lists of flow_y paths for optical flow
    
    def add_vid(self, paths_x, paths_y, paths, gt_a):
        # print("paths: ", paths)
        # print("gt_a: ", gt_a)
        self.videos_flows_X.append(paths_x) # list of frame paths
        self.videos_flows_Y.append(paths_y) # list of frame paths
        self.videos.append(paths)   # list of frame paths
        self.gt_a_list.append(gt_a) # ground truth action label

    def get_rand_vid(self, label, idx=-1):
        """
        Get a random video with the specified label. If idx is specified, return the video at that index.
        """
        match_idxs = []              # list of indices of videos with the specified label
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i) # add index to list if the label matches
        # if the index is specified, return the video at that index
        if idx != -1:
            return self.videos[match_idxs[idx]], self.videos_flows_X[match_idxs[idx]], self.videos_flows_Y[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], self.videos_flows_X[random_idx], self.videos_flows_Y[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        # print("label: ", label)
        # print("self.gt_a_list: ", self.gt_a_list)
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)

"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        print("args: ", args)
        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []
            
        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256)) 
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size)) # # Random 224 × 224 crops are used as augmentation during training

        video_test_list.append(CenterCrop(self.img_size)) # In contrast, only a centre crop is used during evaluation.

        self.transform = {} # apply a series of transformations when Compose is called
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # print("loading {}".format(self.data_dir))
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            
            # When using 'png' based datasets like kinetics, replace 'jpg' to 'png'
            # gets a list of all the file names in the zip file that do not end with '.jpg', that is, the directories.
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
            
                # class/video/jpg_file
                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder: # checks if the video folder for the current image is different from the previous image
                    if len(insert_frames) >= self.seq_len: # enough frames in insert_frames to form a video sequence.
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else: # No that video folder in the train or test list
                            pass
                    #  start accumulating frames for the new video
                    insert_frames = []
                    # label of the video
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]
               
                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id
            # Iteration ends at the last image, so the last video sequence is not added to the database.
            # Add the last video sequence to the database.
            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        else:
            class_folders = os.listdir(self.data_dir)
            # print("class_folders: ", class_folders)
            class_folders.sort()
            class_folders = [c for c in class_folders if c[0] != '.']
            self.class_folders = class_folders
            for class_folder in class_folders:
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                video_folders = [v for v in video_folders]
                # print("video_folders: ", video_folders)
                if self.args.debug_loader:
                    # print("video_folders: ", video_folders[0:1])
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder.lower())
                    # print("c: ", c)
                    # print("video_folder: ", video_folder)


                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder,"img"))
                    flow_x = os.listdir(os.path.join(self.data_dir, class_folder, video_folder,"flow_x"))
                    flow_y = os.listdir(os.path.join(self.data_dir, class_folder, video_folder,"flow_y"))
                    # print("imgs: ", imgs)
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, "img",img) for img in imgs]
                    flow_x.sort()
                    flow_y.sort()
                    flow_x_paths = [os.path.join(self.data_dir, class_folder, video_folder, "flow_x",img) for img in flow_x]
                    flow_y_paths = [os.path.join(self.data_dir, class_folder, video_folder, "flow_y",img) for img in flow_y]
                    paths.sort()
                    flow_x_paths.sort()
                    flow_y_paths.sort()

                    class_id =  class_folders.index(class_folder)
                    c.add_vid(flow_x_paths, flow_y_paths, paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used """
    # whether the video should be added to the training database or the testing database based on the name of the video folder.
    def get_train_or_test_db(self, split=None):

        # print("split: ", split)
        if split is None:
            get_train_split = self.train
        else:
            # print("self.train_test_lists['train'][0]: ", self.train_test_lists["train"][0])
            if split in self.train_test_lists["train"]:
                # print("split: ", split)
                get_train_split = True

            elif split in self.train_test_lists["test"]:
                get_train_split = False

            else:

                return None
            
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            # {train/test}list{03/07}.txt
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            # print("f: ", f)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                # eg: air drumming/-VtLx-mcPds_000012_000022
                # -> air_drumming/-VtLx-mcPds_000012_000022
                data = [x.replace(' ', '_').lower() for x in data]
                # -> air_drumming/-VtLx-mcPds_000012_000022
                data = [x.strip().split(" ")[0] for x in data]
                # -> os.path.split(x) : [air_drumming, -VtLx-mcPds_000012_000022]
                # -> os.path.splitext(os.path.split(x)[1]) : ('-VtLx-mcPds_000012_000022', '')
                # -> os.path.splitext(os.path.split(x)[1])[0] : -VtLx-mcPds_000012_000022
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
            
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists
        # print("lists: ", lists)

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i
    def read_single_image_flow(self, path):
        """read npy"""
        with open(path, 'rb') as f:
            i = np.load(f)
            # center crop 224 x 224
            i = i[8:232, 8:232]
            return i
        
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, paths_flow_x, paths_flow_y, vid_id = c.get_rand_vid(label, idx) 
        n_frames = len(paths)
        n_frames_flow = len(paths_flow_x)
        if n_frames == self.args.seq_len: # default case: 8
            idxs = [int(f) for f in range(n_frames)] # [0, 1, 2, 3, 4, 5, 6, 7]
            idxs_flow = [int(f) for f in range(n_frames_flow)] # [0, 1, 2, 3, 4, 5, 6]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2
    
            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            # idxs_flow = [int(f) for f in range(n_frames_flow)] # 取前7个
            idxs_flow = [int(f) for f in range(n_frames_flow) if f < self.seq_len] # 取前7个

            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        imgs_flow_x = [self.read_single_image_flow(paths_flow_x[i]) for i in idxs_flow]
        imgs_flow_y = [self.read_single_image_flow(paths_flow_y[i]) for i in idxs_flow]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            # img size is 224
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            # imgs shape: [8, 3, 224, 224]
            imgs = torch.stack(imgs)
            imgs_flow_x = [self.tensor_transform(v) for v in imgs_flow_x]
            imgs_flow_x = torch.stack(imgs_flow_x)
            imgs_flow_y = [self.tensor_transform(v) for v in imgs_flow_y]
            imgs_flow_y = torch.stack(imgs_flow_y)
            # combine two flow images to a two-channel image
            imgs_flow = torch.cat((imgs_flow_x, imgs_flow_y), 1)


        return imgs, imgs_flow, vid_id


    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        # print("c: ", c)
        classes = c.get_unique_classes()
        # print("classes: ", classes)
        # print("way: ", self.way)
        # N way K shot
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class # default: 5
        else:
            n_queries = self.args.query_per_class_test # default: 1

        support_set = []
        support_flow_set = []
        support_labels = []
        target_set = []
        target_flow_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            # K shot + N query
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)
            for idx in idxs[0:self.args.shot]:
                vid, flow, vid_id = self.get_seq(bc, idx)
                support_set.append(vid)
                support_flow_set.append(flow)
                support_labels.append(bl)
            for idx in idxs[self.args.shot:]:
                vid, flow, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_flow_set.append(flow)
                target_labels.append(bl)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_flow_set,support_labels))
        random.shuffle(s)
        support_set, support_flow_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_flow_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_flow_set, target_labels, real_target_labels = zip(*t)
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        
        support_flow_set = torch.stack(support_flow_set,dim=0)
        target_flow_set = torch.stack(target_flow_set, dim = 0)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 

        return {"support_set":support_set, "support_flow_set": support_flow_set,"support_labels":support_labels, "target_set":target_set,   "target_flow_set": target_flow_set,"target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}
 

