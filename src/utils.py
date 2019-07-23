import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split
from postprocess import binarize
from tqdm import tqdm
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
        # img = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA) 
        return img#torch.from_numpy(img).float()#.permute([2, 0, 1])
    else:
        img = img/255.0
        # img = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation = cv2.INTER_AREA) 
        return img#torch.from_numpy(img).float().permute([2, 0, 1])

def train_test_split_stratified(df, test_size = 0.2,random_state=42):
    df.drop_duplicates(subset=[config.ID_COLUMN], inplace=True)
    df.loc[df[config.ENCODING_COL]!=-1, "startify"] = 0
    train, val = train_test_split(df, test_size = test_size,random_state=random_state, stratify=df["startify"])
    print("Train: {}, Val: {}".format(train.shape, val.shape))
    return train, val

def mask2rle(img, width=config.WIDTH, height=config.HEIGHT):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
    
def create_submission(meta, predictions, threshold=0.223):
    output = []
    for image_id, mask in tqdm(zip(meta, predictions)):
        # if mask.shape[0] != 1024:
        #     mask = cv2.resize(mask, (1024, 1024), interpolation = cv2.INTER_AREA)
        mask = binarize(mask, threshold)
        rle_encoded = ' '.join(str(rle) for rle in mask2rle(mask))
        output.append([image_id, rle_encoded])

    submission = pd.DataFrame(output, columns=[config.ID_COLUMN, config.ENCODING_COL]).astype(str)
    return submission