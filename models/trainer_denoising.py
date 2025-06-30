import os
import time
import logging
import itertools
import math
import numpy as np
import random
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
import torch.distributions as tdist


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = self.device

        ## init dataloader
        if args.phase == 'train':
            trainset_ = getattr(importlib.import_module('dataloader.dataset_denoising'), args.trainset, None)
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
            self.load_networks('net', self.args.resume)

        if args.rank <= 0:
            logging.info('----- generator parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))

        ## init loss and optimizer
        if args.phase == 'train':
            if args.rank <= 0:
                logging.info('init criterion and optimizer...')
                
            if args.loss_mse:
                self.criterion_mse = nn.MSELoss().to(self.device)
                self.lambda_mse = args.lambda_mse
                if args.rank <= 0:
                    logging.info('  using mse loss...')

            if args.loss_l1:
                self.criterion_l1 = nn.L1Loss().to(self.device)
                self.lambda_l1 = args.lambda_l1
                if args.rank <= 0:
                    logging.info('  using l1 loss...')
            g_params = [self.net.parameters()]
            
            self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(g_params), lr=args.lr)

            if args.resume_optim:
                self.load_networks('optimizer_G', self.args.resume_optim)
            
        
    def set_learning_rate(self, optimizer, epoch):
        current_lr = self.args.lr * 0.3**(epoch//550)
        optimizer.param_groups[0]['lr'] = current_lr
        if self.args.rank <= 0:
            logging.info('current_lr: %f' % (current_lr))


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name', 'coord']:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        if self.args.phase == 'train':
            if self.augmentation:
                # horizontal flip
                if np.random.randint(0, 2) == 1:
                    batch_samples['noisy_img'] = torch.flip(batch_samples['noisy_img'], dims=[2])
                    batch_samples['clean_img'] = torch.flip(batch_samples['clean_img'], dims=[2])

        return batch_samples


    def get_aug_param_torch(self, b=8, numpy=False, camera_type='SonyA7S2'):
        """
        source: https://github.com/megvii-research/PMN/blob/main
        """
        aug_r, aug_g, aug_b = torch.zeros(b), torch.zeros(b), torch.zeros(b)
        r = np.random.randint(2) * 0.25 + 0.25
        u = r
        if np.random.randint(4):

            aug_g = torch.clamp(torch.randn(b) * r, 0, 4*u)
            aug_r = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 0, 4*u)
            aug_b = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 0, 4*u)
    
        daug, _ = torch.min(torch.stack((aug_r, aug_g, aug_b)), dim=0)
        daug[daug>0] = 0
        aug_r = (1+aug_r) / (1+daug) - 1
        aug_g = (1+aug_g) / (1+daug) - 1
        aug_b = (1+aug_b) / (1+daug) - 1
        if numpy:
            aug_r = np.squeeze(aug_r.numpy())
            aug_g = np.squeeze(aug_g.numpy())
            aug_b = np.squeeze(aug_b.numpy())
            
        return aug_r, aug_g, aug_b

    def SNA_torch(self, gt, aug_wb, camera_type='SonyA7S2', ratio=1, black_lr=False, ori=False, iso=None):
        """
        source: https://github.com/megvii-research/PMN/blob/main
        """
        suffix_iso = f'_{iso}' if iso is not None else ''
        p = raw_util.get_camera_noisy_params_max(camera_type + suffix_iso)
        if p is None:
            assert camera_type == 'SonyA7S2'
            camera_type += '_lowISO' if iso<=1600 else '_highISO'
            p = raw_util.get_camera_noisy_params(camera_type=camera_type)
            p['K'] = 0.0009546 * iso * (1 + np.random.uniform(low=-0.01, high=+0.01)) - 0.00193
        else:
            p['K'] = p['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01))
        
        gt = gt * (p['wp'] - p['bl']) / ratio

        aug_wb = torch.from_numpy(aug_wb).to(gt.device)
        dy = gt * aug_wb.reshape(-1,1,1)    
        dn = tdist.Poisson(dy/p['K']).sample() * p['K']
        if black_lr: dy = dy - gt
        dy = dy * ratio / (p['wp'] - p['bl'])
        dn = dn / (p['wp'] - p['bl'])
    
        if ori is False:
            dn *= ratio
        
        return dn, dy

    
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

        self.best_psnr = 0
        self.augmentation = True  # disenable data augmentation to warm up the encoder
        
        for i in range(self.args.start_iter, self.args.max_iter):
            if i > self.args.max_iter // 2:
                self.optimizer_G.param_groups[0]['lr'] = self.args.lr / 2.
            if i > int(self.args.max_iter * 0.8):
                self.optimizer_G.param_groups[0]['lr'] = 1e-5
            logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))
            t0 = time.time()
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                ## prepare data
                batch_samples = self.prepare(batch_samples)
                noisy_img = batch_samples['noisy_img']
                clean_img = batch_samples['clean_img']
                B, C, H, W = noisy_img.shape

                if 'iso' in batch_samples.keys():
                    iso = batch_samples['iso']
                if 'ratio' in batch_samples.keys():
                    ratio = batch_samples['ratio']
                if 'coord' in batch_samples.keys():
                    coord = batch_samples['coord']
                    
                ## apply shot noise augmentation proposed in the PMN paper
                if self.args.use_sna:
                    aug_r, aug_g, aug_b = self.get_aug_param_torch(b=B)
                    aug_wbs = torch.stack((aug_r, aug_g, aug_b, aug_g), dim=1)
                    for bidx in range(B):
                        aug_wb = aug_wbs[bidx].numpy()
                        if np.abs(aug_wb).max() != 0:
                            dn, dy = self.SNA_torch(clean_img[bidx], aug_wb, iso=iso[bidx], ratio=ratio[bidx], black_lr=False,
                                camera_type='SonyA7S2', ori=False)
                            noisy_img[bidx] = noisy_img[bidx] + dn 
                            clean_img[bidx] = clean_img[bidx] + dy

                
                loss = 0
                self.optimizer_G.zero_grad()
                
                output = self.net(noisy_img)
                    
                if self.args.loss_mse:
                    mse_loss = self.criterion_mse(output, clean_img)
                    mse_loss = mse_loss * self.lambda_mse
                    loss += mse_loss
                    log_info += 'mse_loss:%.06f ' % (mse_loss.item())

                if self.args.loss_l1:
                    l1_loss = self.criterion_l1(output, clean_img)
                    l1_loss = l1_loss * self.lambda_l1
                    loss += l1_loss
                    log_info += 'l1_loss:%.06f ' % (l1_loss.item())
            
                ## optimization
                log_info += 'loss_sum:%f ' % (loss.item())
                loss.backward()
                self.optimizer_G.step()

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
                            if self.args.loss_mse:
                                tb_logger.add_scalar('mse_loss', mse_loss.item(), steps)
                            if self.args.loss_l1:
                                tb_logger.add_scalar('l1_loss', l1_loss.item(), steps)
                            

                steps += 1
                
                
            ## visualization
            if i % self.args.vis_freq == 0:
                clean_img_vis = clean_img[0, :-1]
                clean_img_vis = torch.clamp(clean_img_vis, 0, 1)

                noisy_img_vis = noisy_img[0, :-1]
                noisy_img_vis = torch.clamp(noisy_img_vis, 0, 1)

                output_vis = output[0, :-1]
                output_vis = torch.clamp(output_vis, 0, 1)

                img_vis = torch.cat([noisy_img_vis, clean_img_vis, output_vis], dim=-1).permute(1,2,0).detach().cpu().numpy() * 255.
                img_vis = img_vis.astype(np.uint8)
                save_name = os.path.join(self.args.vis_save_dir, 'vis_%d_%d.jpg' % (i, j))
                cv2.imwrite(save_name, img_vis)
                    
            ## save networks
            if i % self.args.save_epoch_freq == 0:
                if self.args.rank <= 0:
                    logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                    self.save_networks('net', i)
                    self.save_networks('optimizer_G', i)
                    

        ## end of training
        if self.args.rank <= 0:
            tb_logger.close()
            self.save_networks('net', 'final')
            logging.info('The training stage is over!!!')


    def test(self):
        save_path = os.path.join(self.args.save_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.net.eval()
        num = 0
        total_time = 0
        
        with torch.no_grad():
            for batch, batch_samples in enumerate(self.test_dataloader):
                batch_samples = self.prepare(batch_samples)
                # noise_gt = batch_samples['noise']
                # iso = batch_samples['iso']
                # ratio = batch_samples['ratio']
                noisy_img = batch_samples['noisy_img']
                clean_img = batch_samples['clean_img']
                B, C, H, W = noisy_img.shape
                
                output = self.net(noisy_img)


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
        if 'optimizer' or 'scheduler' in net_name:
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=strict)


    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
