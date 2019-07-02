import os

DATA_DIR = "../data/siim-png-images"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'train_png')
TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'train_masks')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'test_png')

ENCODING_FILE = os.path.join(DATA_DIR, 'train-rle.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')

ID_COLUMN = 'ImageId'
ENCODING_COL = 'EncodedPixels'

HEIGHT = 128
WIDTH = 128