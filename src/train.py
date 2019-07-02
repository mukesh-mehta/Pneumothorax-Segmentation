import os
import argparse
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import config
from torchsummary import summary
from loader import load_train_val_dataset# add_depth_channel
# from unet_models import UNet11, UNetResNet
from model import UNetResNetV4#, UNetResNetV5, UNetResNetV6, UNet7, UNet8
# from unet_se import UNetResNetSE
# from lovasz_losses import lovasz_hinge, lovasz_softmax
from losses import DiceLoss
# from postprocessing import crop_image, binarize, crop_image_softmax, resize_image
# from metrics import dice_coeff

MODEL_DIR = config.MODEL_DIR
# focal_loss2d = FocalLoss2d()

class CyclicExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, init_lr, min_lr=5e-7, restart_max_lr=1e-5, last_epoch=-1):
        self.gamma = gamma
        self.last_lr = init_lr
        self.min_lr = min_lr
        self.restart_max_lr = restart_max_lr
        super(CyclicExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.last_lr * self.gamma
        if lr < self.min_lr:
            lr = self.restart_max_lr
        self.last_lr = lr
        return [lr]*len(self.base_lrs)

def criterion(logit, truth ):
    # print(type(logit[0]), type(truth))
    loss = DiceLoss()
    return loss(logit[0], truth)

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def train(args):
    print("start training....")
    #load train and val data
    train_loader, val_loader = load_train_val_dataset(batch_size = args.batch_size, num_workers=6)

    model = eval(args.model_name)(args.layers, num_filters=args.nf).cuda()
    # model = eval(args.model_name)(num_filters=args.nf).cuda()
    
    #filename to save models
    if args.exp_name is None:
        model_file = os.path.join(MODEL_DIR, 'best_{}.pth'.format(args.ifold))
    else:
        model_file = os.path.join(MODEL_DIR, args.exp_name, 'best_{}.pth'.format(args.ifold))

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    best_loss = 1
    model.train()
    for epoch in range(args.epochs):
        current_lr = get_lrs(optimizer)
        train_loss = 0
        train_acc = 0
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            image, mask = data
            image = image.type(torch.FloatTensor).cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(mask.cuda()))
            dice  = 1-loss.item()
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            train_acc += dice
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')
            
        val_loss = 0
        for batch_idx, data in data.DataLoader(dataset_val, batch_size = 8, shuffle = False):
            image, mask = data
            image = image.cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(mask.cuda()))
            val_loss+= loss.item()

        _save_ckp = ''
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), model_file)
            _save_ckp = '*'
        

        print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
            val_loss/(batch_idx+1), best_loss, (time.time() - bg) / 60, _save_ckp))
        
        log.info('epoch {}: train loss: {:.4f} best loss: {:.4f} lr: {} {}'
            .format(epoch, train_loss, best_loss, current_lr, _save_ckp))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--ifolds', default='0', type=str, help='kfold indices')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--pad_mode', default='edge', choices=['reflect', 'edge', 'resize'], help='pad method')
    parser.add_argument('--exp_name', default='depths', type=str, help='exp name')
    parser.add_argument('--model_name', default='UNetResNetV4', type=str, help='')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--store_loss_model', action='store_true')
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--meta_version', default=1, type=int, help='meta version')
    parser.add_argument('--pseudo', action='store_true')
    
    args = parser.parse_args()

    print(args)
    ifolds = [int(x) for x in args.ifolds.split(',')]
    print(ifolds)
    log.basicConfig(
        filename = 'trainlog_{}.txt'.format(''.join([str(x) for x in ifolds])), 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)

    for i in ifolds:
        args.ifold = i
        train(args)
