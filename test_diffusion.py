import os
import time
import logging
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from utils.util import setup_logger, print_args
from models.trainer_diffusion import Trainer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='referenceSR Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='test_SID_1004_diffusionv2_wonormalization_longer_genimg_withnorm', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    ## estimation
    parser.add_argument('--noise_param_estm', action='store_true')
    parser.add_argument('--visualize_img', action='store_true')
    parser.add_argument('--visualize_noise', action='store_true')
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--with_camera_settings', action='store_true')
    parser.add_argument('--mean_shift', action='store_true')
    parser.add_argument('--std_shift', action='store_true')
    parser.add_argument('--beta_schedule', default='sigmoid', type=str, help='sigmoid | sigmoid2')
    parser.add_argument('--scale_noise', action='store_true')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--normalize_condition', action='store_true')
    
    
    ## network setting
    parser.add_argument('--net_name', default='UNetAttn', type=str, help='UNet | ')
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--inp_dim', default=4, type=int)
    parser.add_argument('--cond_dim', default=4, type=int)

    
    ## diffusion setting
    parser.add_argument('--diffusion_steps', default=1000, type=int)
    parser.add_argument('--generation_result', default='noise', type=str, help='noise | image')
    parser.add_argument('--self_condition', action='store_true')
    parser.add_argument('--auto_normalize', action='store_true')
    parser.add_argument('--sample_time_range', default='None', type=str)
    parser.add_argument('--diffusion_objective', default='pred_v', type=str)
    parser.add_argument('--use_diffusion_loss_weight', action='store_true')
    parser.add_argument('--min_snr_loss_weight', action='store_true')
    parser.add_argument('--use_noneuniform_sampling', action='store_true')
    parser.add_argument('--use_fourier_feature', action='store_true')
    parser.add_argument('--use_condition_fourier_feature', action='store_true')
    parser.add_argument('--with_timestep_pdf', action='store_true')
    parser.add_argument('--dark_frame', action='store_true')
    

    ## dataloader setting
    parser.add_argument('--testset', default='TestSonyDatasetSingleISO', type=str, help='TestSonyDatasetSingleISO | TestSonyDataset')
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--iso_value', default=250, type=float)
    parser.add_argument('--ratio_value', default=100, type=float)
    
    parser.add_argument('--clean_source', default='SID', type=str, help='SID | DIV2K')
    
    
    
    parser.add_argument('--resume', default='../logs/noise_synthesis/weights/train_SID_1004_diffusionv2_wonormalization_longer_genimg_withnorm/snapshot/net_final.pth', type=str)
    parser.add_argument('--save_folder', default='../logs/noise_synthesis/DIV2K_HR_unprocessed_0910_align', type=str)


    ## setup training environment
    args = parser.parse_args()
    # set_random_seed(args.random_seed)
    

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    args.save_folder = os.path.join(args.save_folder, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
#     log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
#     setup_logger(log_file_path)

    print_args(args)
    cudnn.benchmark = True

    ## test model
    trainer = Trainer(args)
    trainer.test()


if __name__ == '__main__':
    main()
