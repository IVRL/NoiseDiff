"""
modified from: https://github.com/cchen156/Learning-to-See-in-the-Dark/blob/master/test_Sony.py
"""
import glob
import sys
import os
import rawpy
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def vis_raw_file(raw_file, save_path):
    """
    visualize a RAW file into sRGB image.
    """
    if isinstance(raw_file, str):
        raw = rawpy.imread(raw_file)
    else:
        raw = raw_file
        
    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    rgb = np.float32(rgb / 65535.0)
    rgb = rgb * 255
    rgb = rgb.astype(np.uint8)
    
    Image.fromarray(rgb).save(save_path)


if __name__ == "__main__":
    result_dir = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_visrgb"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    in_paths = sorted(glob.glob("/scratch/students/2023-fall-sp-liying/dataset/SID/Sony/short/*"))
    gt_paths = sorted(glob.glob("/scratch/students/2023-fall-sp-liying/dataset/SID/Sony/long/*"))

    for gt_path in gt_paths:
        gt_name = os.path.basename(gt_path)
        gt_save_name = gt_name.replace('.ARW', '.png')

#         vis_raw_file(in_path, os.path.join(result_dir, '%5d_00_%d_noisy.png' % (test_id, ratio)))
        vis_raw_file(gt_path, os.path.join(result_dir, gt_save_name))
        print(os.path.join(result_dir, gt_save_name) + ' saved')