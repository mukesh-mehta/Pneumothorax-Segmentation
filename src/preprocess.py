# https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import rle2mask
# from models import UNetShipV1, UNetShipV2
import config

def create_train_mask_imgs():
	df = pd.read_csv(config.ENCODING_FILE, na_filter=False)
	df = df.groupby('ImageId')['EncodedPixels'].apply(' '.join).reset_index()

	for i, row in tqdm(enumerate(df.values)):
		filename = os.path.join(config.TRAIN_MASK_DIR, row[0]+".png")
		if row[1] != '-1':
			decoded_mask = rle2mask(row[1], config.HEIGHT,config.WIDTH)
			#rgb_mask = cv2.cvtColor(decoded_mask,cv2.COLOR_GRAY2RGB)
			# print(filename)
			cv2.imwrite(filename, decoded_mask)
		else:
			cv2.imwrite(filename, np.zeros((1024,1024),dtype='uint8'))

if __name__ == '__main__':
	create_train_mask_imgs()