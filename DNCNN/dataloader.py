import torch.utils.data as data

import prepare_dataset
import numpy as np
import torch
import common
import random
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))

class NoiseDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return (self.opt['dataroot'].split('/')[-1])


    def __init__(self, opt):
        super(NoiseDataset, self).__init__()
        self.opt = opt
        self.train = opt["train"]
        self.path = opt['dataroot']
        self.imgs = os.listdir(opt['dataroot'])
        self.k_root = opt["kernelroot"]
        self.len_k = len(os.listdir(self.k_root))-1
        self.rgb = opt['rgb_range']
        self.alpha = 1e-6
        self.scale_factor = 4
        self.patch_size = opt["patch_size"]
        self.kernel_size = 19
        self.sigma = random.randrange(0, 40, step=5)
    

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (kernel, None)
        """
        k_idx = random.randint(a=0, b=self.len_k)
        kernel = torch.load(os.path.join(self.k_root,os.listdir(self.k_root)[k_idx]))

        hr, hr_path = self._load_file(idx)
        if self.train:
            hr = self._get_patch(hr)
        hr_tensor = common.im2tensor01(hr).unsqueeze(0)
        hr_tensor = hr_tensor.type(torch.FloatTensor).permute(1, 0, 2, 3)
        kernel = kernel.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        input = F.pad(hr_tensor, pad=(self.kernel_size // 2, self.kernel_size// 2, self.kernel_size // 2,self.kernel_size // 2),
                  mode='circular')
        output = F.conv2d(input, kernel)
        # down-sample
        output = output[:, :, ::self.scale_factor, ::self.scale_factor].permute(1,0,2,3)
        # print(output.shape)

        # add AWGN noise
        noises = np.random.normal(0, self.sigma/255, output.shape)
        noises = torch.from_numpy(noises).type(torch.FloatTensor)
        noise_img = output + noises
        # lr = common.tensor2im(noise_img)
        # plt.imsave('LR_noise.png', lr)
        # sr = common.tensor2im(output)
        # plt.imsave('clean.png', sr)
        return {'input':noise_img.squeeze(0), 'output':noises.squeeze(0), 'clean': output.squeeze(0)}

    def __len__(self):
        return len(self.imgs)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.imgs)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        hr_path = self.imgs[idx]
        hr = common.read_img(osp.join(self.path,hr_path))

        return hr, hr_path

    def _get_patch(self, hr):
        hr = common.get_patch(hr, self.patch_size)
        hr = common.augment([hr])[0]
        return hr 

def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    dataset = NoiseDataset(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset  

def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)