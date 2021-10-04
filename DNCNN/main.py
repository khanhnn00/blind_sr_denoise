import argparse, random
from tqdm import tqdm

import torch

import options.options as option
import util
import os
from solvers import create_solver
from dataloader import create_dataset, create_dataloader
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    opt = option.parse('options/train_ECNN.json')

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)
    loader_list = []

    # create train and val dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase.find('val') == 0:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            loader_list.append(val_loader)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))
        
        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver(opt)
    model_name = opt['networks']['which_model'].upper()
    print(model_name)

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']


    print("Method: %s  || Epoch Range: (%d ~ %d)"%(model_name, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                batch_size = batch['input'].size(0)
                train_loss_list.append(iter_loss*batch_size)
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()

        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))
        
        print('===> Validating...',)

        epoch_is_best = False
        for sth in loader_list:

            val_loss = []
            val_psnr = []
            val_ssim = []
            noise_gt = []
            noise = []
            img_gt = []
            img_pred = []
            input = []
            for iter, batch in enumerate(sth):
                solver.feed_data(batch)
                iter_loss = solver.test()
                val_loss.append(iter_loss)

                # calculate evaluation metrics
                visuals = solver.get_current_visual()
                psnr, ssim = util.calc_metrics(visuals['img_gt'], visuals['img_pred'], test_Y=True)
                val_psnr.append(psnr)
                val_ssim.append(ssim)

                visuals = solver.get_current_visual(need_np=False)

                noise_gt.append(visuals['noise_gt'])
                noise.append(visuals['noise_pred'])
                img_gt.append(np.expand_dims(visuals['img_gt'], axis=0).transpose(0,3,1,2))
                img_pred.append(np.expand_dims(visuals['img_pred'], axis=0).transpose(0,3,1,2))
                input.append(np.expand_dims(visuals['input'], axis=0).transpose(0,3,1,2))

            img_gt = np.concatenate(img_gt, axis=0)
            img_pred = np.concatenate(img_pred, axis=0)
            input = np.concatenate(input, axis=0)
            img_gt = torch.from_numpy(img_gt)
            img_pred = torch.from_numpy(img_pred)
            input = torch.from_numpy(input)

            if opt["save_image"]:
                solver.save_current_visual(epoch, iter, noise_gt, noise, type='noise')
                solver.save_img(epoch, iter, img_gt, img_pred, input)

            solver_log['records']['val_loss'].append(sum(val_loss)/len(val_loss))
            solver_log['records']['psnr'].append(sum(val_psnr)/len(val_psnr))
            solver_log['records']['ssim'].append(sum(val_ssim)/len(val_ssim))
        
            if solver_log['best_pred'] < sum(val_psnr)/len(val_psnr):
                solver_log['best_pred'] = sum(val_psnr)/len(val_psnr)
                epoch_is_best = True
                solver_log['best_epoch'] = epoch
                                                                                                                                             
        print("Loss: %.6f   Best loss: %.2f in Epoch: [%d]" % ((sum(train_loss_list)/len(train_set)),
                                                                                              solver_log['best_pred'],
                                                                                              solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)

    print('===> Finished !')


if __name__ == '__main__':
    main()