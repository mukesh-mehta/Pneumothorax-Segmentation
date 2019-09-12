# https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
from PIL import Image
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

def loss_less_compression(height=config.HEIGHT, width=config.WIDTH):
	df = pd.read_csv(config.ENCODING_FILE, na_filter=False)
	df = df.groupby('ImageId')['EncodedPixels'].apply(' '.join).reset_index()

	for i, row in tqdm(enumerate(df.values)):
		filename = os.path.join(config.TRAIN_IMG_DIR_RAW, row[0]+".png")
		img = Image.open(filename)
		img = img.resize((height,width),Image.ANTIALIAS)
		img.save(os.path.join(config.TRAIN_IMG_DIR_RAW+"{}".format(height), row[0]+".png"))
		
	for i, row in tqdm(enumerate(df.values)):
		filename = os.path.join(config.TRAIN_MASK_DIR_RAW, row[0]+".png")
		img = Image.open(filename)
		img = img.resize((height,width),Image.ANTIALIAS)
		img.save(os.path.join(config.TRAIN_MASK_DIR_RAW+"{}".format(height), row[0]+".png"))

	
def loss_less_compression_test(height=config.HEIGHT, width=config.WIDTH):
	df = pd.read_csv(config.TEST_FILE, na_filter=False)

	for i, row in tqdm(df.iterrows()):
		filename = os.path.join(config.TEST_IMG_DIR_RAW, row[config.ID_COLUMN]+".png")
		img = Image.open(filename)
		img = img.resize((height,width),Image.ANTIALIAS)
		img.save(os.path.join(config.TEST_IMG_DIR_RAW+"{}".format(height), row[0]+".png"))

if __name__ == '__main__':
	create_train_mask_imgs()
	# loss_less_compression()
	# loss_less_compression_test()