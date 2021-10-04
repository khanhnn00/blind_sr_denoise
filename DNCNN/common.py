import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']


####################
# Files & IO
####################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images


def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files


def find_benchmark(dataroot):
    bm_list = [dataroot.find(bm)>=0 for bm in BENCHMARK]
    if not sum(bm_list) == 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name


def read_img(path):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    img = imageio.imread(path, pilmode='RGB')
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

# image processing
# process on numpy image
####################
def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        #if img.shape[2] == 3: # for opencv imread
        #    img = img[:, :, [2, 1, 0]]
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose.copy()).type(torch.FloatTensor).cuda().unsqueeze(0)
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def get_patch(img_tar, patch_size):
    oh, ow = img_tar.shape[:2]

    ip = patch_size
    tp = ip
    ix = random.randrange(0, ow - ip + 1)
    iy = random.randrange(0, oh - ip + 1)
    tx, ty = ix, iy

    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_tar

def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img
