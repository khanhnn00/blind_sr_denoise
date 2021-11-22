import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import tqdm
import os
import matplotlib.pyplot as plt
from .networks import skip, fcn
from .SSIM import SSIM
from scipy.io import savemat


sys.path.append('../')
from util import save_final_kernel_png, get_noise, kernel_shift, move2cpu, tensor2im01

sys.path.append('../../')
from FKP.network import KernelPrior
from .VDNet import VDN
from DNCNN.networks.IRCNN import IRCNN
from DNCNN.loss import TVLoss, AsymLoss
'''
# ------------------------------------------
# models of DIPFKP, etc.
# ------------------------------------------
'''


class DIPFKP:
    '''
    # ------------------------------------------
    # (1) create model, loss and optimizer
    # ------------------------------------------
    '''

    def __init__(self, conf, lr, hr_noise, noise, device=torch.device('cuda')):

        # Acquire configuration
        self.conf = conf
        # print(conf)
        self.lr = lr
        self.hr_noise = hr_noise
        self.sf = conf.sf
        self.kernel_size = min(conf.sf * 4 + 3, 21)

        self.noise = noise.type(torch.FloatTensor).cuda()

        # DIP model
        _, C, H, W = self.lr.size()
        
        self.input_dip = get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        # self.noise_prior = nn.DataParallel(IRCNN()).to(device)
        # for p in self.noise_prior.parameters(): p.requires_grad=True
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.to(device)
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=conf.dip_lr)

        self.net_np = VDN(3, slope=0.2, wf=64, dep_U=4)
        self.net_np = torch.nn.DataParallel(self.net_np).cuda()

        state = torch.load('../DNCNN/experiments/model_state_niidgauss')
        self.net_np.load_state_dict(state)

        # self.net_np = IRCNN()
        # self.net_np = torch.nn.DataParallel(self.net_np).cuda()
        # state = torch.load('../DNCNN/experiments/CBDNet/epochs/best_ckp.pth')
        # self.net_np.load_state_dict(state['state_dict'])
        # for p in self.net_np.parameters(): p.requires_grad = False

        # normalizing flow as kernel prior
        if conf.model == 'DIPFKP':
            # initialze the kernel to be smooth is slightly better
            seed = 5
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True

            self.net_kp = KernelPrior(n_blocks=5, input_size=self.kernel_size ** 2, hidden_size=min((self.sf+1)*5, 25),
                                      n_hidden=1)
            state = torch.load(conf.path_KP)
            self.net_kp.load_state_dict(state['model_state'])
            self.net_kp = self.net_kp.to(device)
            self.net_kp.eval()
            for p in self.net_kp.parameters(): p.requires_grad = True

            self.kernel_code = self.net_kp.base_dist.sample((1, 1)).to(device)
            self.kernel_code.requires_grad = False

            self.optimizer_kp = SphericalOptimizer(self.kernel_size, torch.optim.Adam, [self.kernel_code],
                                                   lr=conf.kp_lr)
            # self.optimizer_kp = torch.optim.Adam([{'params':self.net_kp.parameters()}], lr=conf.kp_lr)
        # loss
        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.laplace_penalty = HyperLaplacianPenalty(3, 0.66).cuda()
        self.tv_loss = TVLoss()

        print('*' * 60 + '\nSTARTED {} on: {}...'.format(conf.model, conf.input_image_path))

    '''
    # ---------------------
    # (2) training
    # ---------------------
    '''

    def train(self):

        # self.optimizer_np = torch.optim.Adam([{'params': self.noise_prior.parameters()}], lr=1e-5)
        # phi_Z = self.net_np(self.lr, 'test')
        # nm = phi_Z[:, :3, ].detach().data
        # dn = self.lr-phi_Z[:, :3, ].detach().data
        # plt.imsave(os.path.join(self.conf.output_dir_path, 'denoise_{}.png'.format(self.conf.img_name)),
        #                                 tensor2im01(dn), vmin=0, vmax=1., dpi=1)
        # plt.imsave(os.path.join(self.conf.output_dir_path, 'noise_map_{}.png'.format(self.conf.img_name)),
        #                                 tensor2im01(nm), vmin=0, vmax=1., dpi=1)
        # print(phi_Z.shape)
        # y, x = phi_Z.shape[2], phi_Z.shape[3]
        # print(phi_Z.shape)
        # print(self.lr.shape)
        # print(lr_image_pad.shape)

        # this_lr = self.lr - phi_Z[:, :3, ]

        # phi_Zz = self.net_np(self.hr_noise, 'test')
        # this_hr = self.hr_noise - phi_Zz[:, :3, ]
        # plt.imsave(os.path.join(self.conf.output_dir_path, 'denoise_test_HR_{}.png'.format(self.conf.img_name)),
        #                                 tensor2im01(this_hr), vmin=0, vmax=1., dpi=1)

        for iteration in tqdm.tqdm(range(600), ncols=60):
            iteration += 1

            self.optimizer_dip.zero_grad()
            self.optimizer_kp.opt.zero_grad()
            # self.optimizer_np.zero_grad()
            # self.optimizer_kp.zero_grad()
            
            '''
            # ---------------------
            # (2.1) forward
            # ---------------------
             '''

            # generate sr image
            sr = self.net_dip(self.input_dip)

            # generate kernel
            kernel, logprob = self.net_kp.inverse(self.kernel_code)
            kernel = self.net_kp.post_process(kernel)

            # blur
            sr_pad = F.pad(sr, mode='circular',
                       pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
            out = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1), groups=3)

            

            # downscale
            out = out[:, :, 0::self.sf, 0::self.sf]

            # add noise

            #groundtruth noise
            out = out + self.noise

            # #not VDNet
            # _, noise = self.net_np(self.lr)
            # out = out + noise

            #VDNet
            # phi_Z = self.net_np(self.lr, 'test')
            # out = out + phi_Z[:, :3, ]
            
            '''
            # ---------------------
            # (2.2) backward
            # ---------------------
             '''
            # freeze kernel estimation, so that DIP can train first to learn a meaningful image
            if iteration <= 75:
                self.kernel_code.requires_grad = False
            else:
                self.kernel_code.requires_grad = True


            # first use SSIM because it helps the model converge faster
            if iteration <= 80:
                loss = 1 - self.ssimloss(out, self.lr)
                # loss += 2e-2 * self.laplace_penalty(sr)
            else:
                loss = self.mse(out, self.lr)
                # loss += 0.003 * self.tv_loss(out)

            loss.backward()
            self.optimizer_dip.step()
            self.optimizer_kp.step()
            # self.optimizer_np.step()

            if (iteration % 200 == 0):
                # phi_Z = self.net_np(sr, 'test')
                # sr = sr - phi_Z[:, :3, ]
                save_final_kernel_png(move2cpu(kernel.squeeze()), self.conf, self.conf.kernel_gt, iteration)
                plt.imsave(os.path.join(self.conf.output_dir_path, '{}_{}.{}'.format(self.conf.img_name, iteration, self.conf.ext)),
                                        tensor2im01(sr), vmin=0, vmax=1., dpi=1)
                plt.imsave(os.path.join(self.conf.output_dir_path, 'LR_{}_{}.{}'.format(self.conf.img_name, iteration, self.conf.ext)),
                                        tensor2im01(out), vmin=0, vmax=1., dpi=1)
                print('\n Iter {}, loss: {}'.format(iteration, loss.data))
        ##save kernel groundtruth
        savemat('%s/k_%s.mat' % (self.conf.output_dir_path, self.conf.img_name.split('.')[0]), {'Kernel': self.conf.kernel_gt.cpu().numpy()})
        kernel = move2cpu(kernel.squeeze())
        save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

        if self.conf.verbose:
            print('{} estimation complete! (see --{}-- folder)\n'.format(self.conf.model,
                                                                         self.conf.output_dir_path) + '*' * 60 + '\n\n')

        return kernel, sr


class SphericalOptimizer(torch.optim.Optimizer):
    ''' spherical optimizer, optimizer on the sphere of the latent space'''

    def __init__(self, kernel_size, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            # in practice, setting the radii as kernel_size-1 is slightly better
            self.radii = {param: torch.ones([1, 1, 1]).to(param.device) * (kernel_size - 1) for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2, param.ndim)), keepdim=True) + 1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss

class HyperLaplacianPenalty(nn.Module):
    def __init__(self, num_channels, alpha, eps=1e-6):
        super(HyperLaplacianPenalty, self).__init__()

        self.alpha = alpha
        self.eps = eps

        self.Kx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
        self.Kx = self.Kx.expand(1, num_channels, 3, 3)
        self.Kx.requires_grad = False
        self.Ky = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda()
        self.Ky = self.Ky.expand(1, num_channels, 3, 3)
        self.Ky.requires_grad = False

    def forward(self, x):
        gradX = F.conv2d(x, self.Kx, stride=1, padding=1)
        gradY = F.conv2d(x, self.Ky, stride=1, padding=1)
        grad = torch.sqrt(gradX ** 2 + gradY ** 2 + self.eps)

        loss = (grad ** self.alpha).mean()

        return loss
