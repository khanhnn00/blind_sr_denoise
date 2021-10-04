import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil
from torchvision.utils import save_image
from loss import TVLoss, AsymLoss

from networks import create_model
from .base_solver import BaseSolver
from networks import init_weights
import util

class ESolver(BaseSolver):
    def __init__(self, opt):
        super(ESolver, self).__init__(opt)
        self.opt = opt
        self.train_opt = opt['solver']
        self.input = self.Tensor()
        self.gt = self.Tensor()
        self.clean = self.Tensor()
        self.pred = None

        self.records = {'train_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'lr': []

        }
        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)
            
            self.criterion_tv = TVLoss().cuda()
            self.criterion_asym = AsymLoss().cuda()

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()
        self.print_network()

        print('===> Solver Initialized : [%s] || Use GPU : [%s]'%(self.__class__.__name__,
                                                                                       self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)


    def feed_data(self, batch, need_HR=True):
        input = batch['input']
        clean = batch['clean']
        self.input.resize_(input.size()).copy_(input)
        self.clean.resize_(clean.size()).copy_(clean)
        # print(self.input.shape)
        self.input.cuda()
        self.clean.cuda()
        if need_HR:
            target = batch['output']
            self.gt.resize_(target.size()).copy_(target)
            self.gt.cuda()
        # print(self.input.shape)
        # print(self.gt.shape)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss_batch = 0.0
        sub_batch_size = int(self.input.size(0) / self.split_batch)
        for i in range(self.split_batch):
            with torch.autograd.set_detect_anomaly(True):
                loss_sbatch = 0.0
                split_LR = self.input.narrow(0, i*sub_batch_size, sub_batch_size)
                split_noise = self.gt.narrow(0, i*sub_batch_size, sub_batch_size)
                split_HR = self.clean.narrow(0, i*sub_batch_size, sub_batch_size)
                output, noise_map = self.model(split_LR)
                # print(output.shape, noise_map.shape)
                loss_sbatch = self.criterion_pix(output, split_HR) + 0.05*self.criterion_tv(noise_map) + 0.5*self.criterion_asym(split_noise, noise_map)

                loss_sbatch /= self.split_batch
                loss_sbatch.backward()

                loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.eval()
        return loss_batch


    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self.model.forward
            SR, noise_map = forward_func(self.input)
            self.pred = SR
            self.noise_map = noise_map

        self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.pred, self.clean)
            return loss_pix.item()
        

    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp'%epoch))

    def save_img(self, epoch, iter, gt, est, inp):
        
        """
        save visual results for comparison
        """
        print('save o: {}'.format(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1))))
        gt_max, _ = gt.flatten(2).max(2, keepdim=True)
        gt = gt / gt_max.unsqueeze(3)
        est_max, _ = est.flatten(2).max(2, keepdim=True)
        est = est / est_max.unsqueeze(3)
        inp_max, _ = inp.flatten(2).max(2, keepdim=True)
        inp = inp / inp_max.unsqueeze(3)
        save_image(gt, os.path.join(self.visual_dir, 'output_img.png' ), nrow=7, normalize=True)
        save_image(est, os.path.join(self.visual_dir, 'pred_img.png' ), nrow=7, normalize=True)
        save_image(inp, os.path.join(self.visual_dir, 'input_img.png' ), nrow=7, normalize=True)

    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.module.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.module.load_state_dict
                load_func(checkpoint)


    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        out_dict['input'] = self.input.data[0].float().cpu()
        out_dict['noise_pred'] = self.noise_map.data[0].float().cpu()
        out_dict['img_pred'] = self.pred.data[0].float().cpu()
        out_dict['img_gt'] = self.clean.data[0].float().cpu()
        print(out_dict['img_pred'].shape, out_dict['img_gt'].shape)
        out_dict['input'], out_dict['img_pred'], out_dict['img_gt'] = util.Tensor2np([out_dict['input'], out_dict['img_pred'], out_dict['img_gt']],
                                                                        self.opt['rgb_range'])
        if need_HR:
            out_dict['noise_gt'] = self.gt.data[0].float().cpu()
            # out_dict['noise_gt'] = util.Tensor2np([out_dict['noise_gt']],
            #                                             self.opt['rgb_range'])[0]
        return out_dict

    def save_current_visual(self, epoch, iter, visual_gt, visual, type='img'):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            for i in range(10):
                visuals_list.extend([util.quantize(visual_gt[i].squeeze(0), self.opt['rgb_range']),
                                    util.quantize(visual[i].squeeze(0), self.opt['rgb_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            if type == 'img':
                misc.imsave(os.path.join(self.visual_dir, 'img_epoch_%d.png' % (epoch)),
                            visual_images)
            else:
                misc.imsave(os.path.join(self.visual_dir, 'noise_epoch_%d.png' % (epoch)),
                            visual_images)


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']

    def save_current_log(self):
        data = {}
        for i in self.records.keys():
            data[i] = self.records[i]
        data_frame = pd.DataFrame(
            data,
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")