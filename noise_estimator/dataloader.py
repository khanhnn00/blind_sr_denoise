import torch.utils.data as data

from data import common
import random
from data import prepare_dataset
import numpy as np
import torch
from noise_estimator import common

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
        self.sigma = [5,10,15,20,25,30,35,40]
        self.paths_HR = img_path
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

        # Adds noise to pixels to dequantize them, ref MAF. This is crucail to add small numbers to zeros of the kernel.
        # No noise will lead to negative NLL, 720 is an empirical value.
        kernel = kernel + np.random.rand(*kernel.shape) / 720.0

        # Transforms pixel values with logit to be unconstrained by np.log(x / (1.0 - x)), [-13.8,13.8], ref MAF
        kernel = logit(self.alpha + (1 - 2 * self.alpha) * kernel)

        kernel = kernel.to(torch.float32)

        hr, hr_path = self._load_file(idx)
        if self.train:
            hr = self._get_patch(hr)
        hr_tensor = common.np2Tensor([hr], self.opt['rgb_range'])

        kernel = kernel.type(torch.FloatTensor).cuda().unsqueeze(0).unsqueeze(0)
        input = F.pad(hr_tensor, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                  mode='circular')
        output = F.conv2d(input, kernel)
        output = output.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()

    # down-sample
    output = output[::scale_factor[0], ::scale_factor[1], :]

    # add AWGN noise
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)

        return kernel, torch.zeros(1)

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
        hr = common.read_img(hr_path, self.opt['data_type'])

        return hr, hr_path


    def _get_patch(self, hr):

        LR_size = self.opt['LR_size']
        hr = common.get_patch(
            hr, LR_size, self.scale)
        hr = common.augment([hr])

        return hr   