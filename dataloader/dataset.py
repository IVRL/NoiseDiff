import os
# from imageio import imread
from PIL import Image, ImageOps
import numpy as np
import glob
import random
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import rawpy
import cv2
import itertools
import sys
sys.path.append('..')
from utils import util
from utils import raw_util

train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
    
# -------------------------------------
# For diffusion model training
# Data resampled for balanced data
# -------------------------------------
class SonyTrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        
        iso_ratio_pairs = util.get_iso_ratio_info()
        iso_ratio_dict = {}
        for pair in iso_ratio_pairs:
            iso, ratio = pair
            key = str(int(iso))+'_'+str(int(ratio))
            iso_ratio_dict[key] = []
        
        
        in_paths = []
        gt_paths = []
        isos = []
        ratios = []
        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    
#                     if iso == iso_value and ratio == ratio_value:
                    index = str(int(iso))+'_'+str(int(ratio))
                    path_pair = [os.path.join(data_folder, in_path), os.path.join(data_folder, gt_path), iso, ratio]
                    iso_ratio_dict[index].append(path_pair)


        img_num = 0
        for value in iso_ratio_dict.values():
            img_num += len(value)
        print('image number: ', img_num)
        
        for item in iso_ratio_dict.items():
            key, value = item
            if len(value) < 100 and len(value) > 0:
                new_value = int(100. / len(value)) * value
                iso_ratio_dict[key] = new_value
                
        sample_list = []
        for value in iso_ratio_dict.values():
            sample_list.extend(value)
        self.sample_num = len(sample_list)
        self.sample_list = sample_list
        print('updated image number: ', self.sample_num)
        
        with open('dataloader/combination_mapping.pickle', 'rb') as handle:
            self.combination_mapping = pickle.load(handle)


    def __len__(self):
        return self.sample_num

    def aug(self, img_list, h, w, phase='train'):
        ih, iw, _ = img_list[0].shape
        if np.random.uniform() < 0.5:
            x = np.random.randint(0, iw - w + 1)
            y = np.random.randint(0, ih - h + 1)
        else:
            x = np.random.randint(0, iw - w + 1)
            y = ih - h - 1

        for i in range(len(img_list)):
            img_list[i] = img_list[i][y:y+h, x:x+w, :]
            
        return img_list

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        in_path, gt_path, iso, ratio = sample
        iso_ratio_idx = self.combination_mapping.get((iso, ratio))
        
        in_fn = os.path.basename(in_path)
        gt_fn = os.path.basename(gt_path)
        
        # read raw images
        raw = rawpy.imread(in_path)
        gt_raw = rawpy.imread(gt_path)

        # subtract the black level and convert to 4-channel images
        input_norm = raw_util.pack_raw(raw) * ratio
        gt_norm = raw_util.pack_raw(gt_raw)
        input_norm = input_norm.clip(0., 1.)  # (H, W, 4)
        noise = (input_norm - gt_norm)  # (H, W, 4)

        
        H, W, C = np.shape(input_norm)
        coord = util.make_coord(H, W, rescale=True).numpy()  # (H, W, 2)

        noise_crop, input_crop, gt_crop, coord_crop = self.aug([noise, input_norm, gt_norm, coord], 
                                                   self.args.crop_size, self.args.crop_size, phase=self.args.phase)
        

        sample = {'noise': noise_crop,
                  'iso': iso,
                  'noisy_img': input_crop,
                  'clean_img': gt_crop,
                  'coord': coord_crop,
                  'iso_ratio_idx': iso_ratio_idx,
                 }

        for key in sample.keys():
            if key not in ['iso', 'iso_ratio_idx']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample
    
    
    
# -------------------------------------
# For diffusion model testing, to generate noise data
# -------------------------------------
class NoiseImageGenerationDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        self.iso_value = iso_value
        self.ratio_value = ratio_value
        
        with open('./pretrained_ckpts/sid_train_clean_info.pickle', 'rb') as handle:
            sid_train_clean_info = pickle.load(handle)

        in_paths = []
        gt_paths = []
        isos = []
        ratios = []
        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    
                    if iso == iso_value and ratio == ratio_value:  # total number: ?
                        in_paths.append(in_path.split('/')[-1])
                        gt_paths.append(gt_path.split('/')[-1])
                        isos.append(iso)
                        ratios.append(ratio)
  
        
        if len(in_paths) >= 20:
            print('Number of clean images is larger than 20!!!')
            sys.exit()
        else:
            clean_img_names = sid_train_clean_info[str(iso_value) + '_' + str(ratio_value)]
            all_clean_img_names = os.listdir(os.path.join(data_folder, 'Sony/long'))
            clean_pool = [item for item in all_clean_img_names if item not in clean_img_names]
            clean_selected = random.sample(clean_pool, 30 - len(in_paths))
            
        gt_paths = [os.path.join(data_folder, 'Sony/long', name) for name in clean_selected]
            
        self.gt_list = gt_paths
        print('image number: ', len(self.gt_list))
        
        self.coord_list = []

        w, h = 4256 // 2, 2848 // 2
        ps = args.crop_size
        step = ps - ps // 4
        thresh_size = ps
        h_space = np.arange(0, h - ps + 1, step)
        if h - (h_space[-1] + ps) < thresh_size:
            h_space = np.append(h_space, h - ps)
        w_space = np.arange(0, w - ps + 1, step)
        if w - (w_space[-1] + ps) < thresh_size:
            w_space = np.append(w_space, w - ps)
        
        
        index = 0
        for y in h_space:
            for x in w_space:
                index += 1
                self.coord_list.append([x, y])
                    
                    
        print('len(self.coord_list) ', len(self.coord_list))
        
        self.patch_per_img = len(self.coord_list)
        self.data_len = len(self.gt_list) * self.patch_per_img
                
        print('data length: ', self.data_len)
        print('img number: ', len(self.gt_list))
        
        with open('dataloader/combination_mapping.pickle', 'rb') as handle:
            self.combination_mapping = pickle.load(handle)


    def __len__(self):
        return self.data_len

    def aug(self, img_list, x, y, h, w):
        for i in range(len(img_list)):
            img_list[i] = img_list[i][y:y+h, x:x+w, :]
        return img_list

    def __getitem__(self, idx):
        img_idx = np.floor(idx / self.patch_per_img).astype(np.uint8)
        gt_path = self.gt_list[img_idx]
        iso = self.iso_value
        ratio = self.ratio_value
        iso_ratio_idx = self.combination_mapping.get((iso, ratio))
        
        gt_fn = os.path.basename(gt_path)
        
        # read raw images
        gt_raw = rawpy.imread(gt_path)

        # subtract the black level and convert to 4-channel images
        gt_norm = raw_util.pack_raw(gt_raw)
        H, W, C = np.shape(gt_norm)
        coord = util.make_coord(H, W, rescale=True).numpy()  # (H, W, 2)
        
        x, y = self.coord_list[idx % self.patch_per_img]
        gt_crop, coord_crop = self.aug([gt_norm, coord],
                                                   x, y,
                                                   self.args.crop_size, self.args.crop_size)
        
        clean_name = os.path.basename(gt_path)

        sample = {
                  'iso': iso,
                  'ratio': ratio,
                  'clean_img': gt_crop,
                  'coord': coord_crop,
                  'clean_name': clean_name,
                  'iso_ratio_idx': iso_ratio_idx,
                  'image_coord': str(int(x))+'_'+str(int(y)),
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'noisy_name', 'clean_name', 'iso_ratio_idx', 'coord_list', 'image_coord']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample
    
    
    

    
    
    



# Ignore
class GenDarkFrameDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value

        in_paths = []
        gt_paths = []
        isos = []
        ratios = []
        iso_ratio_list = []

        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)

                    iso_ratio = str(iso)+'_'+str(int(ratio))

                    if iso_ratio_list.count(iso_ratio) < 1:
                        iso_ratio_list.append(iso_ratio)
                        in_paths.append(os.path.join(data_folder, in_path))
                        gt_paths.append(os.path.join(data_folder, gt_path))
                        isos.append(iso)
                        ratios.append(ratio)
                        
        self.input_list = in_paths#[::-1]
        self.gt_list = gt_paths#[::-1]
        self.iso_list = isos#[::-1]
        self.ratio_list = ratios#[::-1]


        print('image number: ', len(self.input_list))
        
        self.coord_list = []
                    
        w, h = 4256 // 2, 2848 // 2
        ps = args.crop_size
        step = ps - ps // 4
        thresh_size = ps
        h_space = np.arange(0, h - ps + 1, step)
        if h - (h_space[-1] + ps) < thresh_size:
            h_space = np.append(h_space, h - ps)
        w_space = np.arange(0, w - ps + 1, step)
        if w - (w_space[-1] + ps) < thresh_size:
            w_space = np.append(w_space, w - ps)
        
        
        index = 0
        for y in h_space:
            for x in w_space:
                index += 1
                self.coord_list.append([x, y])
                    
                    
        print('len(self.coord_list) ', len(self.coord_list))
        
        self.patch_per_img = len(self.coord_list)
        self.data_len = len(self.input_list) * self.patch_per_img
                
        print('data length: ', self.data_len)
        print('img number: ', len(self.input_list))
        
        with open('dataloader/combination_mapping.pickle', 'rb') as handle:
            self.combination_mapping = pickle.load(handle)


    def __len__(self):
        return self.data_len

    def aug(self, img_list, x, y, h, w):
        for i in range(len(img_list)):
            img_list[i] = img_list[i][y:y+h, x:x+w, :]
        return img_list

    def __getitem__(self, idx):
        img_idx = np.floor(idx / self.patch_per_img).astype(np.uint8)
        in_path = self.input_list[img_idx]
        gt_path = self.gt_list[img_idx]
        iso = self.iso_list[img_idx]
        ratio = self.ratio_list[img_idx]
        iso_ratio_idx = self.combination_mapping.get((iso, ratio))

        W, H = 4256 // 2, 2848 // 2
        coord = util.make_coord(H, W, rescale=True).numpy()  # (H, W, 2)
        
        x, y = self.coord_list[idx % self.patch_per_img]
        coord_crop = self.aug([coord],
                               x, y,
                               self.args.crop_size, self.args.crop_size)
        coord_crop = coord_crop[0]
        
        noisy_name = os.path.basename(in_path)
        clean_name = os.path.basename(gt_path)

        sample = {
                  'iso': iso,
                  'ratio': ratio,
                  'coord': coord_crop,
                  'noisy_name': noisy_name,
                  'clean_name': clean_name,
                  'iso_ratio_idx': iso_ratio_idx,
                  'image_coord': str(int(x))+'_'+str(int(y)),
                  
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'noisy_name', 'clean_name', 'iso_ratio_idx', 'coord_list', 'image_coord',]:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()
            else:
                pass

        return sample
    
    