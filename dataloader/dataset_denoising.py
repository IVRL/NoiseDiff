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
from scipy.stats import truncnorm
import cv2
import itertools
import sys
sys.path.append('..')
from utils import util
from utils import raw_util


train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
synthetic_folder = './NoiseDiff_GeneratedNoiseData'


class SyntheticNoisDiffDenoisingDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        # clean images
        clean_folder = os.path.join(data_folder, 'Sony/long')
        clean_paths = sorted(glob.glob(os.path.join(clean_folder, '*.ARW')))
        clean_imgs = {}
        for clean_path in clean_paths:
            name = os.path.basename(clean_path).split('.ARW')[0]
            clean_raw = rawpy.imread(clean_path)
            clean_norm = raw_util.pack_raw(clean_raw)
            clean_imgs[name] = clean_norm

        # synthetic data of generated SID
        pair_list = []
        for subfolder in os.listdir(synthetic_folder):
            iso_value = subfolder.split('_')[0]
            iso_value = int(iso_value.replace('ISO', ''))

            ratio_value = subfolder.split('_')[1]
            ratio_value = int(ratio_value.replace('Ratio', ''))
            
            
            noise_files = sorted(glob.glob(os.path.join(synthetic_folder, subfolder, '*.npy')))            
            print(subfolder)
            print('noise files in SID: ', len(noise_files))
            for noise_path in noise_files:
                name = os.path.basename(noise_path).split('.npy')[0]
                clean_name, noisy_name, coord = name.split('+')
                pair_list.append([clean_name, noise_path, coord, iso_value, ratio_value])
                    

        self.pair_list = pair_list
        self.clean_imgs = clean_imgs
        print('image number: ', len(self.pair_list))

        # load darkshadings
        self.ds_k_high, self.ds_b_high, self.ds_k_low, self.ds_b_low, self.blc_mean = raw_util.load_darkshading()

    def __len__(self):
        return len(self.pair_list)


    def get_darkshading(self, iso, ds_k, ds_b, blc_mean):
        darkshading = ds_k * iso + ds_b + blc_mean[iso]
        
        return darkshading

    def remove_darkshading(self, raw, iso, ratio, coord):
        x, y = coord.split('_')
        x = int(x); y = int(y)
        x = x * 2; y = y * 2
        
        # unpack raw image to 1 channel
        _, h, w = raw.shape
        H = h * 2; W = w * 2
        raw_unpack = np.zeros((H, W), raw.dtype)
        
        raw_unpack[0:H:2, 0:W:2] = raw[0, :, :]
        raw_unpack[0:H:2, 1:W:2] = raw[1, :, :]
        raw_unpack[1:H:2, 1:W:2] = raw[2, :, :]
        raw_unpack[1:H:2, 0:W:2] = raw[3, :, :]

        raw_unpack = raw_unpack / ratio
        im = raw_unpack * (16383 - 512) + 512
        im = im.clip(0, 16383)

        # subtract the dark shading
        if iso > 1600:
            ds_k = self.ds_k_high
            ds_b = self.ds_b_high
        else:
            ds_k = self.ds_k_low
            ds_b = self.ds_b_low
        darkshading = self.get_darkshading(iso, ds_k, ds_b, self.blc_mean)

        im = im - darkshading[y:y+512*2, x:x+512*2]

        # pack to 4 channels
        im = raw_util.pack_np_raw(im)
        im = np.maximum(im - 512, 0) # subtract the black level
        im = im / (16383 - 512) 
        im = im * ratio
        im = im.clip(0., 1.)
        im = np.transpose(im, (2,0,1))

        return im

    def aug(self, img_list, h, w, phase='train'):
        _, ih, iw = img_list[1].shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, y:y+h, x:x+w]
            
        return img_list

    def __getitem__(self, idx):
        clean_name, noise_path, coord, iso, ratio = self.pair_list[idx] 
        clean_img = self.clean_imgs[clean_name] # (H, W, 4)
        x, y = coord.split('_')
        x = int(x); y = int(y)
        clean_img = clean_img[y:y+512, x:x+512, :]
        clean_img = clean_img.transpose(2,0,1)
        
        noise_img = np.load(noise_path)
        noise_img = np.clip(noise_img, -1., 1.).astype(np.float32)
        noisy_img = noise_img + clean_img
        clean_img = np.clip(clean_img, 0., 1.)
        noisy_img_org = np.clip(noisy_img, 0., 1.)

        if self.args.sub_darkshading:
            noisy_img = self.remove_darkshading(noisy_img_org, iso, ratio, coord)
        else:
            noisy_img = noisy_img_org
        
        noisy_img = np.clip(noisy_img, 0., 1.)

        
        clean_img, noisy_img = self.aug([clean_img, noisy_img],
                                        self.args.crop_size, self.args.crop_size, phase=self.args.phase)

        sample = {'noisy_img': noisy_img,
                  'clean_img': clean_img,
                  'iso': iso,
                  'ratio': ratio
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample
    


class RealSonyDenoisingDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []
        
        # real data
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
                    
                    pair_list.append([os.path.join(data_folder, gt_path), os.path.join(data_folder, in_path), ratio, iso])
                        
            self.data_len = len(pair_list)

        self.pair_list = pair_list
        print('image number: ', len(self.pair_list))

        # load darkshadings
        self.ds_k_high, self.ds_b_high, self.ds_k_low, self.ds_b_low, self.blc_mean = raw_util.load_darkshading()
 
    def __len__(self):
        return self.data_len

    def get_darkshading(self, iso):
        # subtract the dark shading
        if iso > 1600:
            ds_k = self.ds_k_high
            ds_b = self.ds_b_high
        else:
            ds_k = self.ds_k_low
            ds_b = self.ds_b_low
            
        darkshading = ds_k * iso + ds_b + self.blc_mean[iso]
        darkshading = raw_util.pack_np_raw(darkshading).transpose(2,0,1)  # (C, H, W)

        return darkshading

    def aug(self, img_list, h, w, phase='train'):
        _, ih, iw = img_list[1].shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, y:y+h, x:x+w]
            
        return img_list

    def __getitem__(self, idx):
        clean_path, noise_path, ratio, iso = self.pair_list[idx]
        
        raw = rawpy.imread(noise_path)
        gt_raw = rawpy.imread(clean_path)
        clean_img = raw_util.pack_raw(gt_raw, rescale=True)
        noisy_img = raw_util.pack_raw(raw, rescale=False)
        clean_img = clean_img.transpose(2,0,1)
        noisy_img = noisy_img.transpose(2,0,1)
        
        darkshading = self.get_darkshading(iso)
        
        clean_img, noisy_img, darkshading = self.aug([clean_img, noisy_img, darkshading],
                                    self.args.crop_size, self.args.crop_size, phase=self.args.phase)
        
        if self.args.sub_darkshading:
            noisy_img = noisy_img - darkshading

        noisy_img = noisy_img * ratio
        noisy_img = noisy_img.clip(0, 16383 - 512)
        noisy_img = noisy_img / (16383 - 512)

        sample = {'noisy_img': noisy_img,
                  'clean_img': clean_img,
                  'iso': iso,
                  'ratio': ratio
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample

    

    

class PossionGaussianDenoisingDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []
        
        # real clean images
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
                    
                    in_path = os.path.join(data_folder, in_path)
                    gt_path = os.path.join(data_folder, gt_path)

                    pair_list.append([gt_path, iso, ratio])

                        
            self.data_len = len(pair_list)

        self.pair_list = pair_list
        print('image number: ', len(self.pair_list))
        
        # read noise profile
        with open('./pretrained_ckpts/noise_profile_all.pkl', 'rb') as file:
            self.noise_profile = pickle.load(file)
        

    def __len__(self):
        return self.data_len 

    def aug(self, img, h, w, phase='train'):
        _, ih, iw = img.shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        img = img[:, y:y+h, x:x+w]
            
        return img
    

    def generate_truncated_normal(self, mean, variance, lower_bound, upper_bound, sample_size):
        std_dev = np.sqrt(variance)
        a = (lower_bound - mean) / std_dev
        b = (upper_bound - mean) / std_dev 
        truncated_samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=sample_size)

        return truncated_samples


    def apply_noise(self, clean, iso, ratio):
        K, VAR = self.noise_profile[iso]
        latent = clean / float(ratio)
        C, H, W = latent.shape
        latent = latent.reshape(C*H*W)
        k = self.generate_truncated_normal(K, 1, lower_bound=0.7*K, upper_bound=1.3*K, sample_size=1)
        var = self.generate_truncated_normal(VAR, 1, lower_bound=0.7*VAR, upper_bound=1.3*VAR, sample_size=1)
        poisson = k * np.random.poisson(latent / k, size=C*H*W).reshape((C,H, W))
        gaussian = np.random.normal(0, np.sqrt(var), C*H*W).reshape((C,H, W))
        
        noisy = (poisson + gaussian) * ratio
        noisy = noisy.clip(0, 16383 - 512)
        
        return noisy
        

    def __getitem__(self, idx):
        gt_path, iso, ratio = self.pair_list[idx]
                            
        gt_raw = rawpy.imread(gt_path)
        clean_img = raw_util.pack_raw(gt_raw, rescale=False)
        clean_img = clean_img.transpose(2,0,1)

        clean_img = self.aug(clean_img,
                             self.args.crop_size, self.args.crop_size, phase=self.args.phase)
        
        noisy_img = self.apply_noise(clean_img, iso, ratio)
        clean_img = clean_img / (16383 - 512)
        noisy_img = noisy_img / (16383 - 512)

        sample = {
                  'clean_img': clean_img,
                  'noisy_img': noisy_img,
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample
    