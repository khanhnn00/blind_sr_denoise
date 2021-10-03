import torch.utils.data as data

from data import common
import random

class GaussianDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return (self.opt['dataroot_HR'].split('/')[-1])


    def __init__(self, opt):
        super(GaussianDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR= None

    def __getitem__(self, idx):
        hr, hr_path = self._load_file(idx)
        if self.train:
            idx = random.uniform()
        else:
            idx = 
        return {'LR': lr_tensor, 'HR_blur': hr_blur,'HR_path': hr_path}


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
        # random crop and augment
        hr = common.get_patch(
            hr, LR_size, self.scale)
        hr = common.augment([hr])

        return hr   