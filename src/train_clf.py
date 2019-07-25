import os
import argparse
import logging as log
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import config
from torchsummary import summary
from loader import load_train_val_dataset
<<<<<<< HEAD
from metrics import accuracy
# from losses import DiceLoss, FocalLoss2d
=======
from losses import DiceLoss, FocalLoss2d
>>>>>>> adfee714143f28253248c316d5c0d1a583898693
from cnn_finetune import make_model

MODEL_DIR = config.CLF_MODEL_DIR
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

def criterion(logit, truth):
    # print(type(logit[0]), type(truth))
    loss =  nn.BCELoss()
    return loss(nn.Sigmoid()(logit), truth)

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def train(args):
    print("start training....")
    #load train and val data
    train_loader, val_loader = load_train_val_dataset(batch_size = args.batch_size, num_workers=6, dev_mode = args.dev_mode, is_clf=True)

<<<<<<< HEAD
    model = make_model(args.model_name, num_classes=1, pretrained=True, input_size=(1024, 1024)).cuda()

    # print(summary(model, (3, 1024, 1024)))
=======
    model = make_model(args.model_name, num_classes=args.num_classer, pretrained=True, input_size=(config.HEIGHT, config.WIDTH))
    
>>>>>>> adfee714143f28253248c316d5c0d1a583898693
    #filename to save models
    if args.exp_name is None:args.model_name
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
    best_loss = 10
    model.train()
    print("epoch | lr | Progress | batch_loss | loss | batch_acc | acc | val_loss | val_acc | best_loss | time | is_best|")
    for epoch in range(args.epochs):
        current_lr = get_lrs(optimizer)
        train_loss = 0
        total_acc = 0
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
<<<<<<< HEAD

            optimizer.zero_grad()
            image, label = data
            image = image.type(torch.FloatTensor).cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(label.float().view(-1,1).cuda()))
=======
            image, mask = data
            label = np.int32((mask.data.numpy()>0).any())
            del mask
            image = image.type(torch.FloatTensor).cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(label.cuda()))
            dice  = 1-loss.item()
            
            optimizer.zero_grad()
            loss.backward()
>>>>>>> adfee714143f28253248c316d5c0d1a583898693

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = accuracy(y_pred.cpu(),label)
            total_acc += acc
            tot_loss = train_loss/(batch_idx+1)
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), tot_loss, acc, total_acc/(batch_idx+1)), end='')
        
        del loss
        val_acc=0
        val_loss = 0
        for batch_idx, data in enumerate(val_loader):
<<<<<<< HEAD
            image, label = data
            image = image.cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(label.float().view(-1,1).cuda()))
=======
            image, mask = data
            label = np.int32((mask.data.numpy()>0).any())
            del mask
            image = image.cuda()
            y_pred = model(Variable(image))

            loss = criterion(y_pred, Variable(label.cuda()))
>>>>>>> adfee714143f28253248c316d5c0d1a583898693
            val_loss+= loss.item()
            val_acc += accuracy(y_pred.cpu(),label)

        _save_ckp = ''
        loss = val_loss/(batch_idx+1)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), model_file)
            _save_ckp = '*'
        

        print(' {} | {} | {} | {} | {} |'.format(
            val_loss/(batch_idx+1), val_acc/(batch_idx+1),best_loss, (time.time() - bg) / 60, _save_ckp))
        
        log.info('epoch {}: train loss: {:.4f} val acc: {:.4f} best loss: {:.4f} lr: {} {}'
            .format(epoch, tot_loss, val_acc/(batch_idx+1), best_loss, current_lr, _save_ckp))
        del image, label, data, loss, train_loss, total_acc, val_acc, val_loss, tot_loss
        torch.cuda.empty_cache()
        time.sleep(10)
    print(model_file)
    # print("best iou")
    # get_threshold_iou(model, model_file)

def get_threshold_iou(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    model = model.cuda()

    train_loader, val_loader = load_train_val_dataset(batch_size = args.batch_size, num_workers=6, dev_mode = args.dev_mode)
    val_truth=[]
    val_pred=[]
    for batch_idx, data in enumerate(val_loader):
        image, mask = data
        image = image.cuda()
        val_pred.extend(torch.sigmoid(model(Variable(image))).cpu().data.numpy())
        val_truth.extend(mask.cpu().data.numpy())

    val_pred = np.array(val_pred)
    val_truth = np.array(val_truth)
    thresholds = np.linspace(0.2, 0.9, 31)
    ious = np.array([iou_metric_batch(val_truth, np.int32(val_pred > threshold)) for threshold in tqdm(thresholds)])
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    print(threshold_best, iou_best)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--ifolds', default='0', type=str, help='kfold indices')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=4, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--pad_mode', default='edge', choices=['reflect', 'edge', 'resize'], help='pad method')
    parser.add_argument('--exp_name', default='resnet34', type=str, help='exp name')
    parser.add_argument('--model_name', default='resnet34', type=str, help='')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--store_loss_model', action='store_true')
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--meta_version', default=1, type=int, help='meta version')
    parser.add_argument('--pseudo', action='store_true')
    
    args = parser.parse_args()
    args.dev_mode=False
    print(args)
    ifolds = [int(x) for x in args.ifolds.split(',')]
    print(ifolds)
    log.basicConfig(
        filename = 'trainlog_{}.txt'.format(''.join([str(x) for x in ifolds])), 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)
    log.info(args)
    for i in ifolds:
        args.ifold = i
        train(args)