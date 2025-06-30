import os
import sys
# import re
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from torch.nn import init
import torch.nn.functional as F
import functools
import copy
from functools import partial, reduce
import numpy as np
import itertools
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
sys.path.append("..")
from models.attend import Attend

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

    
# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, ks=3, pd=1):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=ks, padding=pd)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, ks=3, pd=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups, ks=3, pd=1)
        self.block2 = Block(dim_out, dim_out, groups=groups, ks=3, pd=1)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
            
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
    
class ResnetBlock2(nn.Module):
    def __init__(self, dim, dim_out, *, pos_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(pos_emb_dim, dim_out * 2, 1, 1, 0)
        ) if exists(pos_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, pos_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(pos_emb):
            pos_emb = self.mlp(pos_emb)
            scale_shift = pos_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class PosEmbedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2=6-1, N_freqs=6,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out
    
    
class LearnedSinusoidalPosEmb(nn.Module):
    """ modified from @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, in_dim, hidden_dim, is_random = False):
        super().__init__()
        self.weights = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0)
        self.out_dim = hidden_dim * 3

    def forward(self, x):
        x = self.weights(x)
        freqs = x * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = 1)
        fouriered = torch.cat((x, fouriered), dim = 1)
        
        return fouriered
    
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
#------------------------------------------------------    
# model
#------------------------------------------------------        
class UNet_PosEmbV2(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        dim = args.dim
        cond_dim = args.cond_dim
        init_dim = None
        out_dim = None
        dim_mults = (1, 2, 4, 8)
        channels = args.inp_dim
        resnet_block_groups = 8
        learned_variance = False
        learned_sinusoidal_cond = False
        random_fourier_features = False
        learned_sinusoidal_dim = 16
        sinusoidal_pos_emb_theta = 10000
        attn_dim_head = 32
        attn_heads = 4
        full_attn = (False, False, False, True)
        flash_attn = True
        self.self_condition = args.self_condition
        self.normalize_condition = args.normalize_condition

        # determine dimensions
        pos_dim = 8

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
        
        # positional encoding
        self.pos_enc = LearnedSinusoidalPosEmb(in_dim=2, hidden_dim=pos_dim)
        self.pos_mlp = Mlp(in_features=pos_dim*3, hidden_features=pos_dim*2, out_features=pos_dim, act_layer=nn.GELU)
        block_klass2 = partial(ResnetBlock2, groups = 2)
        self.pos_block1 = block_klass2(init_dim, init_dim, pos_emb_dim = pos_dim)
        self.pos_block2 = block_klass2(dim, dim, pos_emb_dim = pos_dim)
        
        # condition encoding
        self.cond_init_conv = nn.Conv2d(cond_dim, init_dim, 7, padding = 3)
        self.cond_res_block1 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
#         self.cond_res_block2 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
        self.cond_concat_conv = nn.Conv2d(init_dim*2, init_dim, 3, padding = 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, condition = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        B, C, H, W = x.shape
        clean_img = condition['clean_img']; position = condition['position']
        pos_emb = self.pos_enc(position)  # (B, Cp, H, W)
        pos_emb = self.pos_mlp(pos_emb)
        
        
        clean_emb = self.cond_init_conv(clean_img)
        clean_emb = self.cond_res_block1(clean_emb)
#         clean_emb = self.cond_res_block2(clean_emb)

        x = self.init_conv(x)
        r = x.clone()
        
        x = self.cond_concat_conv(torch.cat([x, clean_emb], dim=1))

        t = self.time_mlp(time)

        h = []
        
        
        x = self.pos_block1(x, pos_emb)

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)

            x = upsample(x)
    
        x = self.pos_block2(x, pos_emb)
        
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    
    
    
class UNet_PosEmbV2_NoPosition(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        dim = args.dim
        cond_dim = args.cond_dim
        init_dim = None
        out_dim = None
        dim_mults = (1, 2, 4, 8)
        channels = args.inp_dim
        resnet_block_groups = 8
        learned_variance = False
        learned_sinusoidal_cond = False
        random_fourier_features = False
        learned_sinusoidal_dim = 16
        sinusoidal_pos_emb_theta = 10000
        attn_dim_head = 32
        attn_heads = 4
        full_attn = (False, False, False, True)
        flash_attn = True
        self.self_condition = args.self_condition
        self.normalize_condition = args.normalize_condition

        # determine dimensions
        pos_dim = 8

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
        
        block_klass2 = partial(ResnetBlock, groups = 2)
        self.pos_block1 = block_klass2(init_dim, init_dim, time_emb_dim = None)
        self.pos_block2 = block_klass2(dim, dim, time_emb_dim = None)
        
        # condition encoding
        self.cond_init_conv = nn.Conv2d(cond_dim, init_dim, 7, padding = 3)
        self.cond_res_block1 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
#         self.cond_res_block2 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
        self.cond_concat_conv = nn.Conv2d(init_dim*2, init_dim, 3, padding = 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, condition = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        B, C, H, W = x.shape
        clean_img = condition
        
        
        clean_emb = self.cond_init_conv(clean_img)
        clean_emb = self.cond_res_block1(clean_emb)
#         clean_emb = self.cond_res_block2(clean_emb)

        x = self.init_conv(x)
        r = x.clone()
        
        x = self.cond_concat_conv(torch.cat([x, clean_emb], dim=1))

        t = self.time_mlp(time)

        h = []
        
        
        x = self.pos_block1(x)

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)

            x = upsample(x)
    
        x = self.pos_block2(x)
        
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    
    
# -----------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=2, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
    

class AttnBlock(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn = CrossAttention(query_dim, context_dim, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, dropout=dropout, glu=True)
        self.proj_out = nn.Conv2d(query_dim, query_dim, 1, 1, 0)
        
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attn(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        
        return x+x_in


class UNet_PosEmbV2_CameraCond(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        dim = args.dim
        cond_dim = args.cond_dim
        init_dim = None
        out_dim = None
        dim_mults = (1, 2, 4, 8)
        channels = args.inp_dim
        resnet_block_groups = 8
        learned_variance = False
        learned_sinusoidal_cond = False
        random_fourier_features = False
        learned_sinusoidal_dim = 16
        sinusoidal_pos_emb_theta = 10000
        attn_dim_head = 32
        attn_heads = 4
        full_attn = (False, False, False, True)
        flash_attn = True
        self.self_condition = args.self_condition
        self.normalize_condition = args.normalize_condition

        # determine dimensions
        pos_dim = 8

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        
        # camera setting condition
        iso_dim = 16
        self.iso_embed = nn.Embedding(100, iso_dim)
        

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                AttnBlock(query_dim=dim_in, context_dim=iso_dim, heads=4, dim_head=32, dropout=0.),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                AttnBlock(query_dim=dim_out, context_dim=iso_dim, heads=4, dim_head=32, dropout=0.),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
        
        # positional encoding
        self.pos_enc = LearnedSinusoidalPosEmb(in_dim=2, hidden_dim=pos_dim)
        self.pos_mlp = Mlp(in_features=pos_dim*3, hidden_features=pos_dim*2, out_features=pos_dim, act_layer=nn.GELU)
        block_klass2 = partial(ResnetBlock2, groups = 2)
        self.pos_block1 = block_klass2(init_dim, init_dim, pos_emb_dim = pos_dim)
        self.pos_block2 = block_klass2(dim, dim, pos_emb_dim = pos_dim)
        
        # condition encoding
        self.cond_init_conv = nn.Conv2d(cond_dim, init_dim, 7, padding = 3)
        self.cond_res_block1 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
#         self.cond_res_block2 = ResnetBlock(init_dim, init_dim, time_emb_dim = None, groups = 8)
        self.cond_concat_conv = nn.Conv2d(init_dim*2, init_dim, 3, padding = 1)
    

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, condition = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        B, C, H, W = x.shape
        clean_img = condition['clean_img']; position = condition['position']
        
        
        # positional condition
        pos_emb = self.pos_enc(position)  # (B, Cp, H, W)
        pos_emb = self.pos_mlp(pos_emb)
        
        # clean image condition
        clean_emb = self.cond_init_conv(clean_img)
        clean_emb = self.cond_res_block1(clean_emb)

        # camera condition
        iso_ratio_idx = condition['iso_ratio_idx']
        iso_emb = self.iso_embed(iso_ratio_idx).unsqueeze(1)  # (B, 1, D)

        # forward
        x = self.init_conv(x)
        r = x.clone()
        
        x = self.cond_concat_conv(torch.cat([x, clean_emb], dim=1))

        t = self.time_mlp(time)

        h = []
        
        
        x = self.pos_block1(x, pos_emb)

        for block1, block2, attn1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            h.append(x)
            
            x = attn1(x, iso_emb)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, attn1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            
            x = attn1(x, iso_emb)

            x = upsample(x)
    
        x = self.pos_block2(x, pos_emb)
        
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

    
    

    

    


if __name__ == "__main__":
    pass