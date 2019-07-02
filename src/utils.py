import numpy as np
import cv2
import torch
from sklearn.model_selection import train_test_split
import config

def rle2mask(rle, height, width):
	mask= np.zeros(width* height)
	array = np.asarray([int(x) for x in rle.split()])
	starts = array[0::2]
	lengths = array[1::2]

	current_position = 0
	for index, start in enumerate(starts):
	    current_position += start
	    mask[current_position:current_position+lengths[index]] = 255
	    current_position += lengths[index]

	return mask.reshape(width, height)


def load_image(path, mask = False):
    img = cv2.imread(str(path))
    if mask:
        img = img[:, :, 0:1] // 255
        img = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA) 
        return torch.from_numpy(img).float()#.permute([2, 0, 1])
    else:
        img = img / 255.0
        img = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA) 
        return torch.from_numpy(img).float().permute([2, 0, 1])

def train_test_split_stratified(df, test_size = 0.1,random_state=42):
	train, val = train_test_split(df, test_size = test_size,random_state=random_state)
	return train, val