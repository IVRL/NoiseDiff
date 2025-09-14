import os
import time
import logging
import math
import argparse
import random
import sys
import numpy as np
from collections import OrderedDict
import pickle
import rawpy
from PIL import Image
import exifread
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# import lpips
from utils.util import setup_logger, print_args
from models.modules import define_G
from utils import raw_util, metric_util
from models.trainer_denoising import Trainer


sid_test_path = '/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_test_list.txt'
sid_eval_path = '/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_val_list.txt'
eld_eval_path = '/scratch/students/2023-fall-sp-liying/code/noise_synthesis/ELD/ELD_official/dataset/Sony_val.txt'
eld_test_path = '/scratch/students/2023-fall-sp-liying/code/noise_synthesis/ELD/ELD_official/dataset/Sony_test.txt'
sid_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
eld_folder = '/scratch/students/2023-fall-sp-liying/dataset/ELD/testset'


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_networks(network, resume, device, strict=True):
    load_path = resume
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path, map_location=torch.device(device))
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'optimizer' or 'scheduler' in net_name:
        network.load_state_dict(load_net_clean)
    else:
        network.load_state_dict(load_net_clean, strict=strict)


def load_all_image_info(image_file, data_folder, iso_value, ratio_value):
    # image_file = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_test_list.txt"
    # data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
    in_paths = []
    gt_paths = []

    with open(image_file, 'r') as file:
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
                
                if iso == iso_value and ratio == ratio_value:
                    in_paths.append(os.path.join(data_folder, in_path))
                    gt_paths.append(os.path.join(data_folder, gt_path))


    return in_paths, gt_paths


def load_image(in_path, gt_path, ratio, iso, ds_correction=True):

    # read raw images
    raw = rawpy.imread(in_path)
    gt_raw = rawpy.imread(gt_path)
    
    # subtract the black level and convert to 4-channel images
    if ds_correction:
        input_norm = raw_util.pack_raw_withdarkshading(raw, iso, ratio) * ratio
    else:
        input_norm = raw_util.pack_raw(raw) * ratio
        
    gt_norm = raw_util.pack_raw(gt_raw)

#     input_norm = np.minimum(input_norm, 1.0)  # (H, W, 4)
    input_norm = np.clip(input_norm, 0., 1.)
    gt_norm = np.clip(gt_norm, 0., 1.)
    
    
    sample = {
              'noisy_img': input_norm,
              'clean_img': gt_norm,
             }

    for key in sample.keys():
        sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        sample[key] = sample[key].permute(2, 0, 1)

    return sample, raw, gt_raw

    

def get_filename_iso():
    def read_sid_txt(filename):
        inp_list = []
        gt_list = []
        iso_list = []
        ratio_list = []
        with open(filename, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    iso_list.append(iso)
                    
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    inp_list.append(in_fn)
                    gt_list.append(gt_fn)
                    
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    ratio_list.append(ratio)
        return inp_list, gt_list, iso_list, ratio_list

    def read_eld_txt(filename):
        inp_list = []
        gt_list = []
        iso_list = []
        ratio_list = []
        with open(filename, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path = line.split(' ')
                    
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path.replace("\n", ""))
                    inp_list.append(in_fn)
                    gt_list.append(gt_fn)
        return inp_list, gt_list            

    def update_eldlist_withiso(sid_path, eld_path):
        sid_inp_list, sid_gt_list, sid_iso_list, sid_ratio_list = read_sid_txt(sid_path)
        eld_inp_list, eld_gt_list = read_eld_txt(eld_path)
        eld_list = []
        for i, eld_inp in enumerate(eld_inp_list):
            idx = sid_inp_list.index(eld_inp)
            eld_list.append([eld_inp, eld_gt_list[i], sid_iso_list[idx], sid_ratio_list[idx]])
            
        return eld_list
            
    eld_eval_list = update_eldlist_withiso(sid_eval_path, eld_eval_path)
    eld_test_list = update_eldlist_withiso(sid_test_path, eld_test_path)

    return eld_eval_list, eld_test_list

    

def read_paired_fns(filename):
    fns = []
    with open(filename, 'r') as file:
        for line in file:
            if line:
                in_path, gt_path, iso, fvalue = line.split(' ')
                iso = int(iso.replace('ISO', ''))
                fns.append((in_path, gt_path, iso))
                    
    # with open(filename) as f:
    #     fns = f.readlines()
    #     fns = [tuple(fn.strip().split(' ')) for fn in fns]
    print('fns', fns)
    sys.exit()
    return fns


def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) #* 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) #* 255.0

    # image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = np.clip(image_numpy, 0, 1)

    return image_numpy


def crop_center(img,cropx,cropy):
    _, _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, :, starty:starty+cropy,startx:startx+cropx]



def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:  # image
        psnr = peak_signal_noise_ratio(Y, X, data_range=data_range)
        # ssim = structural_similarity(Y, X, data_range=data_range, multichannel=True)
        ssim = structural_similarity(Y, X, data_range=data_range, channel_axis=2)
        return {'PSNR':psnr, 'SSIM': ssim}

    else:
        raise NotImplementedError


class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])               
            else:                                     
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)                    
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape        
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)        
        output = num / den * predict
        # print(num / den)

        return output

    

def postprocess_bayer(rawpath, img4c):
    img4c = img4c.detach()
    img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out



def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def process_image_pair(in_path, gt_path, ratio, iso, net, device, corrector, args):
    # Load and prepare the image pair
    save_folder = args.save_folder
    sample, raw, raw_gt = load_image(in_path, gt_path, ratio, iso, ds_correction=args.correct_darkshading)
    
    # Move data to device and add batch dimension
    for key in sample:
        sample[key] = Variable(sample[key].to(device), requires_grad=False)
        sample[key] = sample[key].unsqueeze(0)
    
    noisy_img = sample['noisy_img']
    clean_img = sample['clean_img']
    
    # Network inference
    with torch.no_grad():
        output = net(noisy_img)
    output = output.clamp(0.0, 1.0)
    
    # Illumination correction
    if args.correct_illum:
        output = corrector(output, clean_img)
    
    # Calculate metrics
    output_np = tensor2im(output)
    target = tensor2im(clean_img)
    res = quality_assess(output_np, target, data_range=1)
    
    # Save processed image if requested
    if args.visualize_img:
        image_name = os.path.basename(in_path).split('.ARW')[0]
        output_processed = postprocess_bayer(gt_path, output)
        Image.fromarray(output_processed.astype(np.uint8)).save(os.path.join(save_folder, f"{image_name}_output.png"))
        
        # clean_img = postprocess_bayer(gt_path, clean_img)
        # Image.fromarray(clean_img.astype(np.uint8)).save(os.path.join(args.save_folder, "%s_clean.png"%(image_name)))
        # noisy_img = postprocess_bayer(gt_path, noisy_img)
        # Image.fromarray(noisy_img.astype(np.uint8)).save(os.path.join(args.save_folder, "%s_noisy.png"%(image_name)))
    
    return res


def main():
    parser = argparse.ArgumentParser(description='referenceSR Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    ## estimation
    parser.add_argument('--noise_param_estm', action='store_true')
    parser.add_argument('--visualize_img', action='store_true')
    parser.add_argument('--correct_illum', action='store_true')
    parser.add_argument('--correct_darkshading', action='store_true')
    
    ## network setting
    parser.add_argument('--net_name', default='LSID', type=str, help='')

    ## dataloader setting
    parser.add_argument('--iso', type=int, default=250)
    parser.add_argument('--ratio', type=int, default=300)
    parser.add_argument('--test_dataset', default='SID', type=str, help='SID | ELD')

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save_folder', default='../logs/denoising/inference_withdarkshading', type=str)
    
    
    ## Setup training environment
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    

    ## Setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device

    ## Distributed settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    ## Setup image saving path
    if args.visualize_img:
        args.save_folder = os.path.join(args.save_folder)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    print_args(args)
    cudnn.benchmark = True
    

    ## Init network
    net = define_G(args)
    if args.resume:
        load_networks(net, args.resume, device)
    net.eval()


    
    ## Get test image paths
    if args.test_dataset == 'SID':
        data_folder = sid_folder
        eld_eval_list, eld_test_list = get_filename_iso()
        input_list = eld_eval_list + eld_test_list
        
    elif args.test_dataset == 'ELD':
        databasedir = eld_folder
        scenes = list(range(1, 10+1))
    
        cameras = ['SonyA7S2']     
        suffixes = ['.ARW']
    
        if args.ratio == 100:
            img_ids = [4, 9, 14]
            gt_ids = [6, 11, 16]
        elif args.ratio == 200:
            img_ids = [5, 10, 15]
            gt_ids = [6, 11, 16]
        else:
            raise NotImplementedError
        input_list = list(zip(cameras, suffixes))
            
    else:
        raise NotImplementedError

    
    if args.correct_illum:
        corrector = IlluminanceCorrect()

    test_ratio = args.ratio
    psnr, ssim = [], []

    ## Iterate over test samples
    for img_idx in range(len(input_list)):
        if args.test_dataset == 'SID':
            in_path, gt_path, iso, ratio = input_list[img_idx]
            if ratio != test_ratio:
                continue
                
            in_path = os.path.join(data_folder, 'Sony/short', in_path)
            gt_path = os.path.join(data_folder, 'Sony/long', gt_path)
            res = process_image_pair(
                in_path, gt_path, test_ratio, iso, net, device, 
                corrector, args
            )

            # Record metrics and print results
            psnr.append(res['PSNR'])
            ssim.append(res['SSIM'])
            print(f"Current PSNR: {res['PSNR']}, SSIM: {res['SSIM']}")
            
        elif args.test_dataset == 'ELD':
            camera, suffix = input_list[img_idx]
            for scene_id in scenes:
                scene = f'scene-{scene_id}'
                datadir = os.path.join(databasedir, camera, scene)
                
                for img_id, gt_id in zip(img_ids, gt_ids):
                    in_path = os.path.join(datadir, f'IMG_{img_id:04d}{suffix}')
                    gt_path = os.path.join(datadir, f'IMG_{gt_id:04d}{suffix}')
                    
                    # Compute exposure ratio (unused in loading but kept for clarity)
                    iso_gt, expo_gt = metainfo(gt_path)
                    target_expo = iso_gt * expo_gt
                    iso_in, expo_in = metainfo(in_path)
                    ratio = target_expo / (iso_in * expo_in)  # Not used in load_image
                    
                    res = process_image_pair(
                        in_path, gt_path, test_ratio, iso_in, net, device, 
                        corrector, args
                    )
        
                    # Record metrics and print results
                    psnr.append(res['PSNR'])
                    ssim.append(res['SSIM'])
                    print(f"Current PSNR: {res['PSNR']}, SSIM: {res['SSIM']}")

    print("===> Averaged PSNR: {}, SSIM:{}".format(np.array(psnr).mean(), np.array(ssim).mean()))
    


if __name__ == '__main__':
    main()
