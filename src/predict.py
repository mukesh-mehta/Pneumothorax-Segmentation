import os
import cv2
import pdb
import time
import warnings

import config

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
import segmentation_models_pytorch as smp

from utils import run_length_encode

class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=0):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        if tta==0:
            self.transform = Compose(
                [
                    Normalize(mean=mean, std=std, p=1),
                    Resize(size, size),
                    ToTensor(),
                ]
            )
        else:
            self.transform = Compose(
            [   
                HorizontalFlip(p=1),
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

if __name__ == '__main__':
    size = config.HEIGHT
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_workers = 8
    batch_size = 1
    best_threshold = 0.5
    min_size = 3500
    device = torch.device("cuda:0")
    df = pd.read_csv(config.TEST_FILE)
    testset = DataLoader(
        TestDataset(config.TEST_IMG_DIR_RAW, df, size, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    testset_flipped = DataLoader(
        TestDataset(config.TEST_IMG_DIR_RAW, df, size, mean, std, tta=0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    # model = model_trainer.net # get the model from model_trainer object
    
    encoded_pixels = []
    for j, (batch, batch_flipped) in enumerate(tqdm(zip(testset,testset_flipped))):
        all_model_preds = []
        for i in range(3):
            model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None).cuda()
            model.eval()
            state = torch.load('./model_{}.pth'.format(i), map_location=lambda storage, loc: storage)
            model.load_state_dict(state["state_dict"])

            preds = torch.sigmoid(model(batch.to(device)))
            preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
            pred_flipped = torch.sigmoid(model(batch_flipped.to(device)))
            pred_flipped = pred_flipped.detach().cpu().numpy()[:, 0, :, :]
            preds = np.mean([preds+np.flip(pred_flipped, 2)], 0)
            all_model_preds.append(preds)

        # print(np.array(all_model_preds).shape)
        preds = np.mean(all_model_preds, 0)
        # print(preds.shape)
        # print(all_model_preds[0]==all_model_preds[1],all_model_preds[2]==all_model_preds[1])
        # break
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(probability, best_threshold, min_size)
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv('submission.csv', columns=['ImageId', 'EncodedPixels'], index=False)

    # model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None).cuda()
    # model.eval()
    # state = torch.load('./model_0.pth', map_location=lambda storage, loc: storage)
    # model.load_state_dict(state["state_dict"])
    # encoded_pixels = []
    # for i, (batch, batch_flipped) in enumerate(tqdm(zip(testset,testset_flipped))):
    #     preds = torch.sigmoid(model(batch.to(device)))
    #     preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
    #     # pred_flipped = torch.sigmoid(model(batch_flipped.to(device)))
    #     # pred_flipped = pred_flipped.detach().cpu().numpy()[:, 0, :, :]
    #     # preds = np.mean([preds+np.flip(pred_flipped, 2)], 0)
    #     # print(preds.shape, pred_flipped.shape, np.flip(pred_flipped, axis=1).shape)
    #     # print(preds[0][1][0],np.flip(preds, 2)[0][1][1023])
    #     for probability in preds:
    #         if probability.shape != (1024, 1024):
    #             probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    #         predict, num_predict = post_process(probability, best_threshold, min_size)
    #         if num_predict == 0:
    #             encoded_pixels.append('-1')
    #         else:
    #             r = run_length_encode(predict)
    #             encoded_pixels.append(r)
    # df['EncodedPixels'] = encoded_pixels
    # df.to_csv('submission_ori.csv', columns=['ImageId', 'EncodedPixels'], index=False)