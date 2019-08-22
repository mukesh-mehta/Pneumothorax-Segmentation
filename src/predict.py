import os
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import config
from loader import get_test_loader
from utils import create_submission
from model import UNetResNetV4
from unet_models import UNet11
import segmentation_models_pytorch as smp

def predict(args, model, checkpoint, out_file):
    print('predicting {}...'.format(checkpoint))
    
    model.load_state_dict(torch.load(checkpoint))
    model = model.cuda()

    test_loader = get_test_loader(batch_size = args.batch_size, dev_mode=False)
    meta = test_loader.meta
    outputs = []
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            img = img.type(torch.FloatTensor).cuda()
            output= model(img)
            output = torch.sigmoid(output).cpu().data.numpy()
            outputs.append(output)

            print('{} / {}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
    outputs = np.vstack(outputs)[:, 0, :, :]
    print(outputs.shape)
    submission = create_submission(meta, outputs)
    submission.to_csv(out_file, index=None, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--model_name', default='resnet34',type=str, help='')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    # parser.add_argument('--ifold', required=True, type=int, help='kfold indices')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    # parser.add_argument('--pad_mode', required=True, choices=['reflect', 'edge', 'resize'], help='pad method')
    # parser.add_argument('--exp_name', default='depths', type=str, help='exp name')
    # parser.add_argument('--meta_version', default=1, type=int, help='meta version')
    parser.add_argument('--sub_file', default='all_ensemble_1024.csv', type=str, help='submission file')

    args = parser.parse_args()

    # model = eval(args.model_name)(num_filters=args.nf)
#     model = eval(args.model_name)(args.layers, num_filters=args.nf)
    model = smp.Unet(args.model_name, classes=1, activation='sigmoid', encoder_weights='imagenet').cuda()
    checkpoint = "../data/siim-png-images/models/resnet34_aug_256/best_0.pth"
    out_file = args.sub_file
    #predict_model(args)
    #ensemble_predict(args)
    predict(args, model, checkpoint, out_file)
