import os
import time
import logging
import itertools
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import importlib
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import importlib
import sys
sys.path.append("..")
from utils import util, raw_util 
from models.modules import define_G
from models.losses import PerceptualLoss, AdversarialLoss
from dataloader import DistIterSampler, create_dataloader
from models.denoising_diffusion_pytorch import GaussianDiffusion
import pickle
from ema_pytorch import EMA


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = self.device

        ## init dataloader
        if args.phase == 'train':
            trainset_ = getattr(importlib.import_module('dataloader.dataset'), args.trainset, None)
            self.train_dataset = trainset_(self.args)

            if args.dist:
                dataset_ratio = 1
                train_sampler = DistIterSampler(self.train_dataset, args.world_size, args.rank, dataset_ratio)
                self.train_dataloader = create_dataloader(self.train_dataset, args, train_sampler)
            else:
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        else:
            testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)
            self.test_dataset = testset_(self.args)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        ## init network
        self.net = define_G(args)
        if args.resume:
            self.load_networks('net', self.args.resume, strict=True)
           
        if args.phase == 'train':
            self.ema = EMA(
                        self.net,
                        beta = 0.995,              # exponential moving average factor
                        update_after_step = 500,    # only after this number of .update() calls will it start updating
                        update_every = 20,          # how often to actually update, to save on compute (updates every 10th .update() call)
                    )
            self.ema.to(self.device)


        if args.rank <= 0:
            logging.info('generator parameters: %f' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))
            

        # diffusion model
        self.diffusion = GaussianDiffusion(
            self.net,
            image_size = args.crop_size,
            timesteps = args.diffusion_steps,    # number of steps
            auto_normalize = args.auto_normalize,
            beta_schedule = args.beta_schedule,
            objective = args.diffusion_objective,
            # sampling_timesteps = 100,
            # ddim_sampling_eta = 1.,
        ).to(self.device)

        ## init loss and optimizer
        if args.phase == 'train':
            if args.rank <= 0:
                logging.info('init criterion and optimizer...')
            g_params = [self.net.parameters()]

            self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = CosineAnnealingLR(self.optimizer_G, T_max=args.max_iter)  # T_max=args.max_iter*2

            if args.resume_optim:
                self.load_networks('optimizer_G', self.args.resume_optim)
            if args.resume_scheduler:
                self.load_networks('scheduler', self.args.resume_scheduler)


    def set_learning_rate(self, optimizer, epoch):
        current_lr = self.args.lr * 0.3**(epoch//550)
        optimizer.param_groups[0]['lr'] = current_lr
        if self.args.rank <= 0:
            logging.info('current_lr: %f' % (current_lr))

    
    def vis_results(self, epoch, i, images):
        for j in range(min(images[0].size(0), 5)):
            save_name = os.path.join(self.args.vis_save_dir, 'vis_%d_%d_%d.jpg' % (epoch, i, j))
            temps = []
            for imgs in images:
                temps.append(imgs[j])
            temps = torch.stack(temps)
            B = temps[:, 0:1, :, :]
            G = temps[:, 1:2, :, :]
            R = temps[:, 2:3, :, :]
            temps = torch.cat([R, G, B], dim=1)
            torchvision.utils.save_image(temps, save_name)

    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    
    def prepare(self, batch_samples, augmentation=True):
        for key in batch_samples.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name', 'image_coord']:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        return batch_samples

    
    def train(self):
        if self.args.rank <= 0:
            logging.info('training on ' + self.args.trainset)
            logging.info('%d training samples' % (self.train_dataset.__len__()))
            logging.info('the init lr: %f'%(self.args.lr))
        steps = 0
        self.net.train()

        if self.args.use_tb_logger:
            if self.args.rank <= 0:
                tb_logger = SummaryWriter(log_dir=self.args.save_folder.replace('weights', 'tb_logger'))
                
        for i in range(0, self.args.start_iter):
            self.scheduler.step()
        for i in range(self.args.start_iter, self.args.max_iter):
            self.scheduler.step()
            logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))
            t0 = time.time()
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                    
                ## prepare data
                batch_samples = self.prepare(batch_samples)
                noise_gt = batch_samples['noise']
                iso = batch_samples['iso']
                noisy_img = batch_samples['noisy_img']
                clean_img = batch_samples['clean_img']
                
                if 'lw' in batch_samples.keys():
                    lw_map = batch_samples['lw']
                
                if self.args.with_camera_settings:
                    iso_ratio_idx = batch_samples['iso_ratio_idx']
                if self.args.positional_encoding:
                    coord = batch_samples['coord']
                
                loss = 0
                self.optimizer_G.zero_grad()

                if self.args.generation_result == 'noise':
                    if self.args.with_camera_settings:
                        diffusion_loss = self.diffusion(noise_gt, condition={'clean_img': clean_img, 'iso_ratio_idx': iso_ratio_idx, 'position': coord})
                    elif self.args.positional_encoding:
                        diffusion_loss = self.diffusion(noise_gt, condition={'clean_img': clean_img, 'position': coord, 'lw_map': lw_map})
                    else:
                        diffusion_loss = self.diffusion(noise_gt, condition=clean_img)
                elif self.args.generation_result == 'image':
                    if self.args.with_camera_settings:
                        # logging.info('taking camera settings as condition')
                        diffusion_loss = self.diffusion(noisy_img, condition={'clean_img': clean_img, 'iso_ratio_idx': iso_ratio_idx, 'position': coord})
                    elif self.args.positional_encoding:
                        # logging.info('training with positional encoding')
                        diffusion_loss = self.diffusion(noisy_img, condition={'clean_img': clean_img, 'position': coord})
                    else:
                        diffusion_loss = self.diffusion(noisy_img, condition=clean_img)

                
                loss = loss + diffusion_loss
                log_info += 'diffusion_loss:%.06f ' % (diffusion_loss.item())
            
                ## optimization
                log_info += 'loss_sum:%f ' % (loss.item())
                loss.backward()
                self.optimizer_G.step()
                
                self.ema.update()
                

                ## print information
                if j % self.args.log_freq == 0:
                    t1 = time.time()
                    log_info += '%4.6fs/batch' % ((t1-t0)/self.args.log_freq)
                    if self.args.rank <= 0:
                        logging.info(log_info)
                    t0 = time.time()
                    

                ## write tb_logger
                if self.args.use_tb_logger:
                    if steps % self.args.vis_step_freq == 0:
                        if self.args.rank <= 0:
                            tb_logger.add_scalar('diffusion_loss', diffusion_loss.item(), steps)
                            tb_logger.add_scalar('lr', self.optimizer_G.param_groups[0]['lr'], steps)

                steps += 1


            ## save networks
            if i % self.args.save_epoch_freq == 0:
                if self.args.rank <= 0:
                    logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                    self.save_networks('net', i)
                    self.save_networks('ema', i)
                    # self.save_networks('optimizer_G', i)
                    # self.save_networks('scheduler', i)

        ## end of training
        if self.args.rank <= 0:
            tb_logger.close()
            self.save_networks('net', 'final')
            self.save_networks('ema', 'final')
            logging.info('The training stage is over!!!')

    
    def normalize_to_zero_to_one(self, x, y):
        min_value = min(x.min(), y.min())
        max_value = max(x.max(), y.max())
        x = (x - min_value) / (max_value - min_value)
        y = (y - min_value) / (max_value - min_value)
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        return x, y


    def test(self):
        save_path = os.path.join(self.args.save_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        img_save_path = os.path.join(save_path, 'img')
        if not os.path.exists(img_save_path):
            os.mkdir(img_save_path)
            
        npy_save_path = os.path.join(save_path, 'npy')
        if not os.path.exists(npy_save_path):
            os.mkdir(npy_save_path)
            

        self.net.eval()
        reg_num = 0
        num = 0
        npy_num = 0

        total_time = 0
        
        with torch.no_grad():
            for batch, batch_samples in enumerate(self.test_dataloader):
                batch_samples = self.prepare(batch_samples, augmentation=False)
                if 'clean_img' in batch_samples.keys():
                    clean_img = batch_samples['clean_img']
                    clean_name = batch_samples['clean_name']
                    
                coord = batch_samples['coord']
                image_coord = batch_samples['image_coord']
                noisy_name = None
                if 'noisy_img' in batch_samples.keys():
                    noisy_img = batch_samples['noisy_img']
                    noisy_name = batch_samples['noisy_name']
                if 'iso_ratio_idx' in batch_samples.keys():
                    iso_ratio_idx = batch_samples['iso_ratio_idx']
                if 'noise' in batch_samples.keys():
                    noise_gt = batch_samples['noise']

                if 'iso' in batch_samples.keys():
                    iso = batch_samples['iso']
                if 'ratio' in batch_samples.keys():
                    ratio = batch_samples['ratio']
                    
                
                # B, C, H, W = clean_img.shape
                B, C, H, W = coord.shape
                
   
                if self.args.with_camera_settings:
                    if self.args.positional_encoding: 
                        if not self.args.dark_frame:
                            output = self.diffusion.sample(batch_size=self.args.batch_size, 
                                                           condition={'clean_img': clean_img, 'iso_ratio_idx': iso_ratio_idx, 'position': coord})
                        else:
                            output = self.diffusion.sample(batch_size=self.args.batch_size, 
                                                           condition={'clean_img': torch.zeros((B, 4, H, W), dtype=coord.dtype, 
                                                                        device=coord.device), 'iso_ratio_idx': iso_ratio_idx, 'position': coord})
                    else:
                        output = self.diffusion.sample(batch_size=self.args.batch_size, 
                                                       condition={'clean_img': clean_img, 'iso_ratio_idx': iso_ratio_idx, 'position': torch.zeros_like(coord)})
                    

                        
                if self.args.save_npy:
                    clean_npy_folder = os.path.join(npy_save_path, 'clean')
                    noisy_npy_folder = os.path.join(npy_save_path, 'noisy')
                    output_npy_folder = os.path.join(npy_save_path, 'generated')

                    if not os.path.exists(clean_npy_folder):
                        os.makedirs(clean_npy_folder)
                    if not os.path.exists(noisy_npy_folder):
                        os.makedirs(noisy_npy_folder)
                    if not os.path.exists(output_npy_folder):
                        os.makedirs(output_npy_folder)

                    for i in range(coord.shape[0]):
                        if not self.args.dark_frame:
                            image_coord_i = image_coord[i]
                            clean_name_i = clean_name[i].split('.ARW')[0]
                            if noisy_name is not None:
                                save_name = noisy_name[i].split('.ARW')[0]
                            else:
                                save_name = clean_name[i].split('.ARW')[0]
                            save_name = clean_name_i + '+' + save_name + '+'+ image_coord_i + '.npy'
                            np.save(os.path.join(output_npy_folder, save_name), output[i].detach().cpu().numpy())
                        else:
                            image_coord_i = image_coord[i]
                            save_name = '%05d'%npy_num
                            save_name = save_name + '_' + str(int(iso[i])) + '_' + str(int(ratio[i])) + '+' + image_coord_i + '.npy'
                            print(save_name)
                            np.save(os.path.join(output_npy_folder, save_name), output[i].detach().cpu().numpy())
        
                        npy_num += 1
                    

    def save_image(self, tensor, path):
        img = Image.fromarray(((tensor/2.0 + 0.5).data.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
        img.save(path)

        
    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device(self.device))
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        
        for k, v in load_net.items():
            
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
                

        network.load_state_dict(load_net_clean, strict=strict)


    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        if net_name == 'ema':
            network = network.ema_model
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
