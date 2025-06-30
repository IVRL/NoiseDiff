import os
import sys
import time
from datetime import datetime
import logging
import numpy as np
import torch
import math

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def setup_logger(log_file_path):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_file_path)


def print_args(args):
    for arg in vars(args):
        logging.info(arg + ':%s'%(getattr(args, arg)))
        
        
# ---- functions -----
def tempsigmoid(x, temp=1.):
    y = torch.sigmoid(x/(temp))
    y =  y * 2 - 1
    return y


def inversesigmoid(x, temp=1.):
    y = (x + 1) / 2
    y = torch.log(y) - torch.log(1 - y)
    y = y * temp
    return y


def piecewise_mapping(x, b=0.2, temp=0.1):
    """
    if x >= -b and x <= b:
        y = tempsigmoid(x, temp)
    elif x < -b:
        y = k2 * x + k2 - 1
    elif x > b:
        y = k2 * x + 1 - k2
    """
    
    assert b < 1 and b > 0, "b should be in the range of (-1,1)"
    k2 = (tempsigmoid(torch.tensor(b), temp) - 1) / (b - 1)
    y = torch.zeros_like(x)
    y[torch.logical_and(x >= -b, x <= b)] = tempsigmoid(x[torch.logical_and(x >= -b, x <= b)], temp)
    y[x < -b] = k2 * x[x < -b] + k2 - 1
    y[x > b] = k2 * x[x > b] + 1 - k2
    
    return y


def inverse_piecewise_mapping(y, b=0.2, temp=0.1):
    """
    if y > b * k1:
        x = (y + k2 - 1) / k2
    elif y < -b * k1:
        x = (y - k2 + 1) / k2
    else:
        x = inversesigmoid(x, temp)
    """
    value_at_b = tempsigmoid(torch.tensor(b), temp)
    k2 = (value_at_b - 1) / (b - 1)
    x = torch.zeros_like(y)
        
    x[y > value_at_b] = (y[y > value_at_b] + k2 - 1) / k2
    x[y < -value_at_b] = (y[y < -value_at_b] - k2 + 1) / k2
    x[torch.logical_and(y <= value_at_b, y >= -value_at_b)] = inversesigmoid(y[torch.logical_and(y <= value_at_b, y >= -value_at_b)], temp)
    
    return x
    

    
def make_coord(h, w, rescale=False):
    seq1 = torch.arange(h).float()
    seq2 = torch.arange(w).float()
    if rescale:
        seq1 = seq1 / (h - 1)
        seq2 = seq2 / (w - 1)
    coord = torch.stack(torch.meshgrid([seq1, seq2], indexing='ij'), dim=-1)
    

    return coord


def get_iso_ratio_info():
    train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list_modified.txt"
    data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
    info_list = []

    i = 0

    with open(train_path, 'r') as file:
        for line in file:
            if line:
                i += 1
                in_path, gt_path, iso, fvalue = line.split(' ')
                iso = int(iso.replace('ISO', ''))

                in_fn = os.path.basename(in_path)
                gt_fn = os.path.basename(gt_path)
                test_id = int(in_fn[0:5])
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)

                info_list.append([iso, ratio])


    print('total number:', i)
    arr_tuples = [tuple(row) for row in info_list]
    info_list = np.unique(arr_tuples, axis=0)

#     print('info_list: ', info_list)
#     print(len(info_list))
    
    return info_list



#  ------ for KLD evaluation ---------------
# adopted from: https://github.com/BorealisAI/noise_flow/blob/master/sidd/sidd_utils.py

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers


def kl_div_forward(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))


def kl_div_inverse(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(q * np.log(q / p))


def kl_div_sym(p, q):
    return (kl_div_forward(p, q) + kl_div_inverse(p, q)) / 2.0


def kl_div_3(p, q):
    kl_fwd = kl_div_forward(p, q)
    kl_inv = kl_div_inverse(p, q)
    kl_sym = (kl_inv + kl_fwd) / 2.0
    return kl_fwd, kl_inv, kl_sym


def kldiv_patch_set(i, mb, x_samples, sc_sd, subdir, klds_que):
    y = unpack_raw(mb['_y'][i, :, :, :])
    nlf_sd = np.sqrt(mb['nlf0'] * y + mb['nlf1'])  # Camera NLF
    ng = np.random.normal(0, sc_sd, y.shape)  # Gaussian
    ns = unpack_raw(x_samples[i, :, :, :])  # NF-sampled
    nl = nlf_sd * np.random.normal(0, 1, y.shape)  # Camera NLF
    n = unpack_raw(mb['_x'][i, :, :, :])  # Real
    xs = np.clip(y + ns, 0.0, 1.0)
    xg = np.clip(y + ng, 0.0, 1.0)
    xl = np.clip(y + nl, 0.0, 1.0)
    x = np.clip(y + n, 0.0, 1.0)
    pid = mb['pid'][i]
    noise_pats_raw = (ng, nl, ns, n)

    # histograms
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    hists = [None] * len(noise_pats_raw)
    klds = np.ndarray([len(noise_pats_raw)])
    klds[:] = 0.0
    for h in reversed(range(len(noise_pats_raw))):
        hists[h], bin_centers = get_histogram(noise_pats_raw[h], bin_edges=bin_edges)
        # noinspection PyTypeChecker
        klds[h] = kl_div_forward(hists[-1], hists[h])

    klds_que.put(klds)



if __name__ == "__main__":
    h=4;w=10
    coord = make_coord(h, w, rescale=True)
    print(coord.shape)
    print(coord[:, :, 1])    