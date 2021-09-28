import os
import argparse
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from util import read_image, im2tensor01, map2tensor, tensor2im01, analytic_kernel, kernel_shift, evaluation_dataset, tensor2im
from config.configs import Config
from model.model import DIPFKP
import random
from data.prepare_dataset import my_degradation, degradation
from torchvision.utils import save_image

# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet

'''
# ------------------------------------------------ 
# main.py for DIP-KP
# ------------------------------------------------
'''


def train(conf, lr_image):
    ''' trainer for DIPFKP, etc.'''
    model = DIPFKP(conf, lr_image)
    kernel, sr = model.train()
    return kernel, sr


def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--path_KP', os.path.abspath(args.path_KP),
              '--sf', args.sf]
    if args.SR:
        params.append('--SR')
    if args.real:
        params.append('--real')
    return params


def main():
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default='DIPFKP', help='models: DIPFKP, DIPSoftmax, DoubleDIP.')
    prog.add_argument('--dataset', '-d', type=str, default='Set5',
                      help='dataset, e.g., Set5.')
    prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')
    prog.add_argument('--path-nonblind', type=str, default='../data/pretrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')
    prog.add_argument('--real', action='store_true', default=False, help='if the input is real image')
    prog.add_argument('--noise', type=int, default=0, help='if the input is real image')

    # to be overwritten automatically
    prog.add_argument('--path-KP', type=str, default='../data/result/log_FKP/FKP_x4/best_model_checkpoint.pt',
                      help='path for trained kernel prior')
    prog.add_argument('--kernel-dir', type=str, default='../data/result/datasets/Kernel_validation_set_x4',
                      help='path for trained kernel prior')
    prog.add_argument('--input-dir', '-i', type=str, default='../../SRbenchmark/HR_x4',
                      help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str,
                      default='../data/log_KernelGANFKP/Set5_DIPFKP_lr_x2', help='path to image output directory')


    args = prog.parse_args()

    # overwritting paths
    args.path_KP = '../data/result/log_FKP/FKP_x4/best_model_checkpoint.pt'
    # args.path_KP = '../data/pretrained_models/FKP_x4.pt'
    args.input_dir = '../../SRbenchmark/HR_x4'
    args.output_dir = '../data/log_DIPFKP/{}_{}_3lr_x{}'.format(args.dataset, args.model, args.sf)
    args.kernel_dir = '../data/result/datasets/Kernel_validation_set_x4'
    k_list = os.listdir(args.kernel_dir)

    # load nonblind model
    if args.SR:
        netG = USRNet(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                      nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        netG.load_state_dict(torch.load(args.path_nonblind), strict=True)
        netG.eval()
        for key, v in netG.named_parameters():
            v.requires_grad = False
        netG = netG.cuda()

    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()
    for filename in filesource[:]:
        print(filename)

        # kernel estimation
        k_idx = random.randint(a=0, b=len(k_list)-1)
        k = torch.load(os.path.join(args.kernel_dir, k_list[k_idx]))
        conf = Config(filename, k).parse(create_params(filename, args))
        # k_idx = random.randint(a=0, b=len(k_list)-1)
        # k = torch.load(os.path.join(args.kernel_dir, k_list[k_idx]))
        save_image(k.unsqueeze(0), os.path.join(conf.output_dir_path, './k_GT.png'),nrow=1,  normalize=True)
        lr_image = im2tensor01(read_image(os.path.join(args.input_dir, filename))).unsqueeze(0)
        # print(lr_image.shape)
        lr_image = my_degradation(lr_image, k, 4, args.noise)
        lr = lr_image.copy()
        lr_image = im2tensor01(lr_image).unsqueeze(0)
        plt.imsave(os.path.join(conf.output_dir_path, '%s_LR.png' % conf.img_name), lr)
        # print(lr_image.size())

        # crop the image to 960x960 due to memory limit
        if 'DIV2K' in args.input_dir:
            crop = int(960 / 2 / conf.sf)
            lr_image = lr_image[:, :, lr_image.shape[2] // 2 - crop: lr_image.shape[2] // 2 + crop,
                       lr_image.shape[3] // 2 - crop: lr_image.shape[3] // 2 + crop]

        kernel, sr_dip = train(conf, lr_image)
        plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr_dip), vmin=0,
                   vmax=1., dpi=1)

        # nonblind SR
        if args.SR:
            kernel = map2tensor(kernel)

            sr = netG(lr_image, torch.flip(kernel, [2, 3]), int(args.sf),
                      (10 if args.real else 0) / 255 * torch.ones([1, 1, 1, 1]).cuda())
            plt.imsave(os.path.join(conf.output_dir_path, '%s.png' % conf.img_name), tensor2im01(sr), vmin=0,
                       vmax=1., dpi=1)

    if not conf.verbose:
        evaluation_dataset(args.input_dir, conf)

    prog.exit(0)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
