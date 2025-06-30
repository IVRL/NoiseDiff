import sys
import os
import pickle
import numpy as np
import exifread
import torch  
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import matplotlib.pyplot as plt
import rawpy
from PIL import Image

resources_path = '/scratch/students/2023-fall-sp-liying/code/noise_synthesis'


def pack_raw(raw, rescale=True):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) # subtract the black level
    if rescale:
        im = im / (16383 - 512) 
#         im = im.clip(0., 1.)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :], # r
                          im[0:H:2, 1:W:2, :], # g
                          im[1:H:2, 1:W:2, :], # b
                          im[1:H:2, 0:W:2, :]), axis=2) # g

    return out



def pack_np_raw(im):
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :], # r
                          im[0:H:2, 1:W:2, :], # g
                          im[1:H:2, 1:W:2, :], # b
                          im[1:H:2, 0:W:2, :]), axis=2) # g
    return out 


def pack_raw_withoutclip(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = im / 16383  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def unpack_raw(raw):
    # unpack raw image to 1 channel
    h, w, _ = raw.shape
    H = h * 2; W = w * 2
    raw_unpack = np.zeros((H, W), raw.dtype)
    
    raw_unpack[0:H:2, 0:W:2] = raw[:, :, 0]
    raw_unpack[0:H:2, 1:W:2] = raw[:, :, 1]
    raw_unpack[1:H:2, 1:W:2] = raw[:, :, 2]
    raw_unpack[1:H:2, 0:W:2] = raw[:, :, 3]

    out = raw_unpack * (16383 - 512) + 512
    out = out.astype(np.uint16)
    out = out.clip(0, 16383)

    return out

    
def get_darkshading(iso):
    with open(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading_BLE.pkl'), 'rb') as f:
        blc_mean = pickle.load(f)
    branch = '_highISO' if iso > 1600 else '_lowISO'
    ds_k = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_k.npy' % branch), allow_pickle=True)
    ds_b = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_b.npy' % branch), allow_pickle=True)
    darkshading = ds_k * iso + ds_b + blc_mean[iso]
    return darkshading


def load_darkshading():
    with open(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading_BLE.pkl'), 'rb') as f:
        blc_mean = pickle.load(f)

    branch = '_highISO'
    ds_k_high = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_k.npy' % branch), allow_pickle=True)
    ds_b_high = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_b.npy' % branch), allow_pickle=True)

    branch = '_lowISO'
    ds_k_low = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_k.npy' % branch), allow_pickle=True)
    ds_b_low = np.load(os.path.join(resources_path, './LRD/LRD_official/resources', 'darkshading%s_b.npy' % branch), allow_pickle=True)

    return ds_k_high, ds_b_high, ds_k_low, ds_b_low, blc_mean


def pack_raw_withdarkshading(raw, iso, ratio):
    # to align with training
    im = raw.raw_image_visible.astype(np.float32)
    im = (im - 512) / (16383 - 512)
    im = (im * ratio).clip(0,1)

    im = im / ratio
    im = im * (16383 - 512) + 512
    im = im.clip(0, 16383)

    # subtract darkshading
    im = im - get_darkshading(iso)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # pack Bayer image to 4 channels
    out = np.concatenate((im[0:H:2, 0:W:2, :], # r
                          im[0:H:2, 1:W:2, :], # g
                          im[1:H:2, 1:W:2, :], # b
                          im[1:H:2, 0:W:2, :]), axis=2) # g

    out = np.maximum(out - 512, 0) # subtract the black level
    out = out / (16383 - 512) 

    return out


def extract_iso_from_arw(arw_file_path):
    try:
        # Open the ARW image file for reading its EXIF metadata
        with open(arw_file_path, 'rb') as file:
            tags = exifread.process_file(file)
            
            # Check if the 'EXIF ISOSpeedRatings' tag is present
            if 'EXIF ISOSpeedRatings' in tags:
                iso_speed = str(tags['EXIF ISOSpeedRatings'])
                iso_speed = int(iso_speed)
                return iso_speed
            
            # If 'EXIF ISOSpeedRatings' tag is not found, return None or handle as needed
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def sliding_window(x, kernel_size=3, dilation=1, stride=1):
    B, C, H, W = x.shape
    patch = F.unfold(x, kernel_size=(kernel_size, kernel_size), padding=dilation, 
                     stride=stride, dilation=dilation)  # [B, C*ks*ks, Hr*Wr]
    patch = patch.view(B, C, kernel_size*kernel_size, -1)
    return patch

# patch-based
def compute_poisson_lambda_by_patch(x):
    B, C, H, W = x.shape
    patch = sliding_window(x)  # [B, C, ks*ks, N]
    std, mean = torch.std_mean(patch, dim=2)  # [B, C, N]
    std = std.view(B*C, -1).cpu().numpy()  # [B*C, N]
    mean = mean.view(B*C, -1).cpu().numpy()
    
    lambda_list = []
    intercept_list = []
    for i in range(B*C):
        X = mean[i].reshape(-1, 1)  # [N, 1]
        y = std[i].reshape(-1, 1)
    
        reg = LinearRegression().fit(X, y)
        lambda_list.append(reg.coef_[0][0])
        intercept_list.append(reg.intercept_)

    lambda_list = np.array(lambda_list).reshape(B, C)
    intercept_list = np.array(intercept_list).reshape(B, C)
    
    return lambda_list, intercept_list


# def compute_poisson_lambda_single_image(x, visualize=False, savepath=''):
#     H, W = x.shape
#     x = x.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
#     patch = sliding_window(x)  # [1, 1, ks*ks, N]
#     std, mean = torch.std_mean(patch, dim=2)  # [1, 1, N]
#     std = std.view(-1).cpu().numpy()  # [N,]
#     mean = mean.view(-1, 1).cpu().numpy()  # [N, 1]

#     X = mean; y = std
#     reg = TheilSenRegressor().fit(X, y)
#     lambda_ = reg.coef_[0]
#     sigma_ = reg.intercept_
#     if visualize:
#         plt.scatter(X, y,color='g', s=0.3)
#         plt.plot(X, reg.predict(X),color='k')
#         plt.savefig(savepath)
#         plt.clf()
    
#     return lambda_, sigma_


# patch-based
def compute_poisson_lambda_by_patch_single_image(x, visualize=False, savepath=''):
    C, H, W = x.shape
    x = x.unsqueeze(0)  # [1, C, H, W]
    patch = sliding_window(x)  # [1, C, ks*ks, N]
    std, mean = torch.std_mean(patch, dim=2)  # [1, C, N]
    std = std.view(-1).cpu().numpy()  # [C*N]
    mean = mean.view(-1, 1).cpu().numpy()  # [C*N, 1]

    X = mean; y = std
    reg = TheilSenRegressor().fit(X, y)
    lambda_ = reg.coef_[0]
    sigma_ = reg.intercept_
    if visualize:
        plt.scatter(X, y,color='g', s=0.3)
        plt.plot(X, reg.predict(X),color='k')
        plt.savefig(savepath)
        plt.clf()
    
    return lambda_, sigma_




def kl_divergence(input, target):
    kl_loss = nn.KLDivLoss(reduction="mean")
    
    input = F.log_softmax(input.view(-1), dim=0)
    target = F.softmax(target.view(-1), dim=0)
    result = kl_loss(input, target)

    return result


# value-based
def get_poisson_lambda(clean, noisy, visualize=False, savepath=''):
    """
    clean: [C, H, W]
    noisy: [C, H, W]
    """
    unique_values = torch.unique(clean)
    means = []
    stds = []
    median_value = torch.median(unique_values)
    for value in unique_values:
        if value <= median_value:
            # position = torch.nonzero(clean==value, as_tuple=True)
            position = torch.nonzero(torch.abs(clean-value)<1e-6, as_tuple=True)
            points = noisy[position[0], position[1], position[2]]
            std = torch.std(points)
            if ~torch.isnan(std):
                stds.append(std.cpu().numpy())
                means.append(value.cpu().numpy())
    
    X = np.array(means).reshape(-1, 1); y = np.array(stds)
    if len(X) > 0:
        reg = TheilSenRegressor().fit(X, y)
        lambda_ = reg.coef_[0]
        sigma_ = reg.intercept_
        if visualize:
            plt.scatter(X, y,color='g', s=0.3)
            plt.plot(X, reg.predict(X),color='k')
            plt.savefig(savepath)
            plt.clf()

        return lambda_, sigma_
    else:
        return 0, 0
    
    
# value-based
def get_poisson_lambda_all_images(clean, noisy, mean_std_dict):
    """
    clean: [C, H, W]
    noisy: [C, H, W]
    """
    unique_values = torch.unique(clean)
    for value in unique_values:
        # position = torch.nonzero(clean==value, as_tuple=True)
        position = torch.nonzero(torch.abs(clean-value)<1e-6, as_tuple=True)
        points = noisy[position[0], position[1], position[2]]
        pints = list(points)
        if value in mean_std_dict.keys():
            mean_std_dict[value].extend(points)
        else:
            mean_std_dict[value] = points
            
    return mean_std_dict


def get_regression_result_all_images(mean_std_dict, visualize=False, savepath=''):
    stds = []
    means = []
    for mean in mean_std_dict.keys():
        std = torch.std(mean_std_dict[mean])
        if ~torch.isnan(std):
            stds.append(std.cpu().numpy())
            means.append(mean.cpu().numpy())
    
    X = np.array(means).reshape(-1, 1); y = np.array(stds)
    reg = TheilSenRegressor().fit(X, y)
    lambda_ = reg.coef_[0]
    sigma_ = reg.intercept_
    if visualize:
        plt.scatter(X, y,color='g', s=0.3, alpha=0.05)
        plt.plot(X, reg.predict(X),color='k')
        plt.savefig(savepath)
        plt.clf()
    
    return lambda_, sigma_

def modify_raw_file(raw_file, tab, position, out_file=''):
    """Takes an ARW or DNG file (uncompressed, with mosaic)
    and writes the tab array (uint16) into it
    starting at position pos
    the visible variable is unused, because in a Sony SID
    the invisible part of the image is a strip at the end of the image
    so the positions are the same
    with other devices, there may be a difference
    between raw_image_visible and positions in raw_image
    The resulting file is "out_file" """
    
    raw = rawpy.imread(raw_file)
    (l,c) = raw.raw_image.shape
    f = open(raw_file,'rb')
    t = f.read()
    f.close()
    tabraw = np.frombuffer(t[-l*c*2:], dtype=np.uint16).reshape((l, c)).copy()
    header = t[:-l*c*2]
    tabraw[position[0] : position[0]+tab.shape[0], position[1] : position[1]+tab.shape[1]] = tab

    # write file
    f = open(out_file,'wb')
    f.write(header)
    f.write(tabraw)
    f.close()
    
    
def vis_raw_file(raw_file, save_path, save_file=True):
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
    rgb = np.clip(rgb, 0, 255)
    
    
#     cmin = 0; cmax = 255; low = 0; high = 255
#     rgb = (rgb*255 - cmin)*(high - low)/(cmax - cmin) + low
    
    if save_file:
        rgb = rgb.astype(np.uint8)
        Image.fromarray(rgb).save(save_path)
    
    return rgb
    
    
    
def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['NikonD850'] = {
        'Kmin':1.2, 'Kmax':2.4828, 'lam':-0.26, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigTLk':0.906, 'sigTLb':-0.6754,   'sigTLsig':0.035165,
        'sigRk':0.8322,  'sigRb':-2.3326,   'sigRsig':0.301333,
        'sigGsk':0.8322, 'sigGsb':-0.1754,  'sigGssig':0.035165,
    }
    cam_noisy_params['IMX686'] = { # ISO-640~6400
        'Kmin':-0.19118, 'Kmax':2.16820, 'lam':0.102, 'q':1/(2**10), 'wp':1023, 'bl':64,
        'sigTLk':0.85187, 'sigTLb':0.07991,   'sigTLsig':0.02921,
        'sigRk':0.87611,  'sigRb':-2.11455,   'sigRsig':0.03274,
        'sigGsk':0.85187, 'sigGsb':0.67991,   'sigGssig':0.02921,
    }
    cam_noisy_params['SonyA7S2_lowISO'] = {
        'Kmin':-1.67214, 'Kmax':0.42228, 'lam':-0.026, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigRk':0.78782,  'sigRb':-0.34227,  'sigRsig':0.02832,
        'sigTLk':0.74043, 'sigTLb':0.86182, 'sigTLsig':0.00712,
        'sigGsk':0.82966, 'sigGsb':1.49343, 'sigGssig':0.00359,
        'sigReadk':0.82879, 'sigReadb':1.50601, 'sigReadsig':0.00362,
        'uReadk':0.01472, 'uReadb':0.01129, 'uReadsig':0.00034,
    }
    cam_noisy_params['SonyA7S2_highISO'] = {
        'Kmin':0.64567, 'Kmax':2.51606, 'lam':-0.025, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigRk':0.62945,  'sigRb':-1.51040,  'sigRsig':0.02609,
        'sigTLk':0.74901, 'sigTLb':-0.12348, 'sigTLsig':0.00638,
        'sigGsk':0.82878, 'sigGsb':0.44162, 'sigGssig':0.00153,
        'sigReadk':0.82645, 'sigReadb':0.45061, 'sigReadsig':0.00156,
        'uReadk':0.00385, 'uReadb':0.00674, 'uReadsig':0.00039,
    }
    cam_noisy_params['CRVD'] = {
        'Kmin':1.31339, 'Kmax':3.95448, 'lam':0.015, 'q':1/(2**12), 'wp':4095, 'bl':240,
        'sigRk':0.93368,  'sigRb':-2.19692,  'sigRsig':0.02473,
        'sigGsk':0.95387, 'sigGsb':0.01552, 'sigGssig':0.00855,
        'sigTLk':0.95495, 'sigTLb':0.01618, 'sigTLsig':0.00790
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']


    
    
def get_camera_noisy_params_max(camera_type=None):
    cam_noisy_params = {
        'SonyA7S2_50': {'Kmax': 0.047815, 'lam': 0.1474653, 'sigGs': 1.0164667, 'sigGssig': 0.005272454, 'sigTL': 0.70727646, 'sigTLsig': 0.004360543, 'sigR': 0.13997398, 'sigRsig': 0.0064381803, 'bias': 0, 'biassig': 0.010093017, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_64': {'Kmax': 0.0612032, 'lam': 0.13243394, 'sigGs': 1.0509665, 'sigGssig': 0.008081373, 'sigTL': 0.71535635, 'sigTLsig': 0.0056863446, 'sigR': 0.14346549, 'sigRsig': 0.006400559, 'bias': 0, 'biassig': 0.008690166, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_80': {'Kmax': 0.076504, 'lam': 0.1121489, 'sigGs': 1.180899, 'sigGssig': 0.011333668, 'sigTL': 0.7799473, 'sigTLsig': 0.009347968, 'sigR': 0.19540153, 'sigRsig': 0.008197397, 'bias': 0, 'biassig': 0.0107246125, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
        'SonyA7S2_100': {'Kmax': 0.09563, 'lam': 0.14875287, 'sigGs': 1.0067395, 'sigGssig': 0.0033682834, 'sigTL': 0.70181876, 'sigTLsig': 0.0037532174, 'sigR': 0.1391465, 'sigRsig': 0.006530218, 'bias': 0, 'biassig': 0.007235429, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_125': {'Kmax': 0.1195375, 'lam': 0.12904578, 'sigGs': 1.0279676, 'sigGssig': 0.007364685, 'sigTL': 0.6961967, 'sigTLsig': 0.0048687346, 'sigR': 0.14485553, 'sigRsig': 0.006731584, 'bias': 0, 'biassig': 0.008026363, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_160': {'Kmax': 0.153008, 'lam': 0.094135, 'sigGs': 1.1293099, 'sigGssig': 0.008340453, 'sigTL': 0.7258587, 'sigTLsig': 0.008032158, 'sigR': 0.19755602, 'sigRsig': 0.0082754735, 'bias': 0, 'biassig': 0.0101351, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_200': {'Kmax': 0.19126, 'lam': 0.07902429, 'sigGs': 1.2926387, 'sigGssig': 0.012171176, 'sigTL': 0.8117464, 'sigTLsig': 0.010250768, 'sigR': 0.22815849, 'sigRsig': 0.010726711, 'bias': 0, 'biassig': 0.011413908, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_250': {'Kmax': 0.239075, 'lam': 0.051688068, 'sigGs': 1.4345995, 'sigGssig': 0.01606571, 'sigTL': 0.8630922, 'sigTLsig': 0.013844714, 'sigR': 0.26271912, 'sigRsig': 0.0130637, 'bias': 0, 'biassig': 0.013569083, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_320': {'Kmax': 0.306016, 'lam': 0.040700804, 'sigGs': 1.7481371, 'sigGssig': 0.019626873, 'sigTL': 1.0334468, 'sigTLsig': 0.017629284, 'sigR': 0.3097104, 'sigRsig': 0.016202712, 'bias': 0, 'biassig': 0.017825918, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_400': {'Kmax': 0.38252, 'lam': 0.0222538, 'sigGs': 2.0595572, 'sigGssig': 0.024872316, 'sigTL': 1.1816813, 'sigTLsig': 0.02505812, 'sigR': 0.36209714, 'sigRsig': 0.01994737, 'bias': 0, 'biassig': 0.021005306, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_500': {'Kmax': 0.47815, 'lam': -0.0031342343, 'sigGs': 2.3956928, 'sigGssig': 0.030144656, 'sigTL': 1.31772, 'sigTLsig': 0.028629242, 'sigR': 0.42528257, 'sigRsig': 0.025104137, 'bias': 0, 'biassig': 0.02981831, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_640': {'Kmax': 0.612032, 'lam': 0.002566592, 'sigGs': 2.9662898, 'sigGssig': 0.045661453, 'sigTL': 1.6474211, 'sigTLsig': 0.04671843, 'sigR': 0.48839623, 'sigRsig': 0.031589635, 'bias': 0, 'biassig': 0.10000693, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_800': {'Kmax': 0.76504, 'lam': -0.008199721, 'sigGs': 3.5475867, 'sigGssig': 0.052318197, 'sigTL': 1.9346539, 'sigTLsig': 0.046128694, 'sigR': 0.5723769, 'sigRsig': 0.037824076, 'bias': 0, 'biassig': 0.025339302, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_1000': {'Kmax': 0.9563, 'lam': -0.021061005, 'sigGs': 4.2727833, 'sigGssig': 0.06972333, 'sigTL': 2.2795107, 'sigTLsig': 0.059203167, 'sigR': 0.6845563, 'sigRsig': 0.04879781, 'bias': 0, 'biassig': 0.027911892, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_1250': {'Kmax': 1.195375, 'lam': -0.032423194, 'sigGs': 5.177596, 'sigGssig': 0.092677385, 'sigTL': 2.708437, 'sigTLsig': 0.07622563, 'sigR': 0.8177013, 'sigRsig': 0.06162229, 'bias': 0, 'biassig': 0.03293372, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_1600': {'Kmax': 1.53008, 'lam': -0.0441045, 'sigGs': 6.29925, 'sigGssig': 0.1153261, 'sigTL': 3.2283993, 'sigTLsig': 0.09118158, 'sigR': 0.988786, 'sigRsig': 0.078567736, 'bias': 0, 'biassig': 0.03877672, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_2000': {'Kmax': 1.9126, 'lam': -0.012963797, 'sigGs': 2.653871, 'sigGssig': 0.015890995, 'sigTL': 1.4356787, 'sigTLsig': 0.02178686, 'sigR': 0.33124214, 'sigRsig': 0.018801652, 'bias': 0, 'biassig': 0.01570677, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_2500': {'Kmax': 2.39075, 'lam': -0.027097283, 'sigGs': 3.200225, 'sigGssig': 0.019307792, 'sigTL': 1.6897862, 'sigTLsig': 0.025873765, 'sigR': 0.38264316, 'sigRsig': 0.023769397, 'bias': 0, 'biassig': 0.018728448, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_3200': {'Kmax': 3.06016, 'lam': -0.034863412, 'sigGs': 3.9193838, 'sigGssig': 0.02649232, 'sigTL': 2.0417721, 'sigTLsig': 0.032873377, 'sigR': 0.44543457, 'sigRsig': 0.030114045, 'bias': 0, 'biassig': 0.021355819, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_4000': {'Kmax': 3.8252, 'lam': -0.043700505, 'sigGs': 4.8015847, 'sigGssig': 0.03781628, 'sigTL': 2.4629273, 'sigTLsig': 0.042401053, 'sigR': 0.52347374, 'sigRsig': 0.03929801, 'bias': 0, 'biassig': 0.026152484, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_5000': {'Kmax': 4.7815, 'lam': -0.053150143, 'sigGs': 5.8995814, 'sigGssig': 0.0625814, 'sigTL': 2.9761007, 'sigTLsig': 0.061326735, 'sigR': 0.6190265, 'sigRsig': 0.05335372, 'bias': 0, 'biassig': 0.058574405, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_6400': {'Kmax': 6.12032, 'lam': -0.07517104, 'sigGs': 7.1163535, 'sigGssig': 0.08435366, 'sigTL': 3.4502964, 'sigTLsig': 0.08226275, 'sigR': 0.7218788, 'sigRsig': 0.0642334, 'bias': 0, 'biassig': 0.059074216, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_8000': {'Kmax': 7.6504, 'lam': -0.08208357, 'sigGs': 8.916516, 'sigGssig': 0.12763213, 'sigTL': 4.269624, 'sigTLsig': 0.13381928, 'sigR': 0.87760293, 'sigRsig': 0.07389065, 'bias': 0, 'biassig': 0.084842026, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_10000': {'Kmax': 9.563, 'lam': -0.073289566, 'sigGs': 11.291476, 'sigGssig': 0.1639773, 'sigTL': 5.495318, 'sigTLsig': 0.16279395, 'sigR': 1.0522343, 'sigRsig': 0.094359785, 'bias': 0, 'biassig': 0.107438326, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_12800': {'Kmax': 12.24064, 'lam': -0.06495205, 'sigGs': 14.245901, 'sigGssig': 0.17283991, 'sigTL': 7.038261, 'sigTLsig': 0.18822834, 'sigR': 1.2749791, 'sigRsig': 0.120479785, 'bias': 0, 'biassig': 0.0944684, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_16000': {'Kmax': 15.3008, 'lam': -0.060692135, 'sigGs': 17.833515, 'sigGssig': 0.19809262, 'sigTL': 8.877547, 'sigTLsig': 0.23338738, 'sigR': 1.5559287, 'sigRsig': 0.15791349, 'bias': 0, 'biassig': 0.09725099, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_20000': {'Kmax': 19.126, 'lam': -0.060213074, 'sigGs': 22.084776, 'sigGssig': 0.21820943, 'sigTL': 11.002351, 'sigTLsig': 0.28806436, 'sigR': 1.8810822, 'sigRsig': 0.18937257, 'bias': 0, 'biassig': 0.4984733, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        'SonyA7S2_25600': {'Kmax': 24.48128, 'lam': -0.09089118, 'sigGs': 25.853043, 'sigGssig': 0.35371417, 'sigTL': 12.175712, 'sigTLsig': 0.4215717, 'sigR': 2.2760193, 'sigRsig': 0.2609267, 'bias': 0, 'biassig': 0.37568903, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}
        }
    cam_noisy_params['IMX686_6400'] = {
        'Kmax':8.74253, 'sigGs':12.8901, 'sigGssig':0.03,
        'sigTL':12.8901, 'lam':0.015, 'sigR':0,
        'q':1/(2**10), 'wp':1023, 'bl':64, 'bias':-0.56896687
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        # log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}".''')
        return None        
    
    
if __name__ == "__main__":
    import os
    import glob
    
    # # extract iso info
    # iso_list = []
    # files = glob.glob('/scratch/students/2023-fall-sp-liying/dataset/SID/Sony/short/*.ARW')
    # for arw_file_path in files:
    #     iso_speed = extract_iso_from_arw(arw_file_path)
    #     iso_list.append(iso_speed)
    #     if iso_speed:
    #         print(f"ISO Speed: %d" % iso_speed)
    #     else:
    #         print("ISO information not found in the EXIF metadata.")
    # unique_iso = list(set(iso_list))
    # print(unique_iso)


    # try to use modify_raw_file
    files = glob.glob('/scratch/students/2023-fall-sp-liying/dataset/SID/Sony/long/*.ARW')
    raw_file = files[1]
    input_name = os.path.basename(raw_file).split('.')[0]
    output_name = input_name + '_modified'
    out_file = output_name + '.ARW'

    raw = rawpy.imread(raw_file)
    raw_pack = pack_raw(raw)
    raw_recon = unpack_raw(raw_pack)
    position = (0, 0)
    modify_raw_file(raw_file, raw_recon, position, output_name+'.ARW')
    raw_recon = rawpy.imread(out_file)
    
    vis_raw_file(out_file, output_name+'.png')
    vis_raw_file(raw_file, input_name+'.png')





