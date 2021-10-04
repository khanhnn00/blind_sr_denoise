import torch.utils.data as data

from data import common
import random
from data import prepare_dataset
import numpy as np
import torch
from DNCNN import common
import torch.nn.functional as F

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
        return (self.opt['dataroot_HR'].split('/')[-1])


    def __init__(self, img_path):
        super(NoiseDataset, self).__init__()
        self.paths_HR = img_path
        self.scale_factor = 4
        self.kernel_size = 19
        self.sigma = random.randrange(0, 40, step=5)
        self.normalization = round(prepare_dataset.gen_kernel_fixed(np.array([self.kernel_size, self.kernel_size]),
                                                                    np.array([self.scale_factor, self.scale_factor]),
                                                                    0.175 * self.scale_factor,
                                                                    0.175 * self.scale_factor, 0,
                                                                    0).max(), 5) + 0.01

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (kernel, None)
        """
        kernel = prepare_dataset.gen_kernel_random(np.array([self.kernel_size, self.kernel_size]),
                                                    np.array([self.scale_factor, self.scale_factor]),
                                                    0.175 * self.scale_factor, min(2.5 * self.scale_factor, 10), self.noise)
        kernel = torch.from_numpy(kernel)

        # Normalization
        kernel = torch.clamp(kernel, min=0) / self.normalization
        kernel = kernel + np.random.rand(*kernel.shape) / 720.0
        kernel = logit(self.alpha + (1 - 2 * self.alpha) * kernel)
        kernel = kernel.to(torch.float32)

        hr, hr_path = self._load_file(idx)
        if self.train:
            hr = self._get_patch(hr)
        hr_tensor = common.np2Tensor([hr], 1)[0].permute(1, 0, 2, 3)

        kernel = kernel.type(torch.FloatTensor).cuda().unsqueeze(0).unsqueeze(0)
        input = F.pad(hr_tensor, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                  mode='circular')
        output = F.conv2d(input, kernel)

        # down-sample
        output = output[:, :, ::self.scale_factor, ::self.scale_factor].permute(1,0,2,3)

        # add AWGN noise
        noises = np.random.normal(0, self.sigma/255, output.shape)
        noises = torch.from_numpy(noises).type(torch.FloatTensor).cuda()
        noise_img = output + noises

        return {'input':noise_img, 'output':noises}

    def __len__(self):
        return len(self.paths_HR)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        hr_path = self.paths_HR[idx]
        hr = common.read_img(hr_path)

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