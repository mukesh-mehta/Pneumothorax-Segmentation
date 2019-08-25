import os

HEIGHT = 1024
WIDTH = 1024

DATA_DIR = "../data/siim-png-images"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR_RAW = os.path.join(TRAIN_DIR, 'train_png')
TRAIN_MASK_DIR_RAW =  os.path.join(TRAIN_DIR, 'train_masks')
TEST_IMG_DIR_RAW = os.path.join(TEST_DIR, 'test_png')

ENCODING_FILE = os.path.join(DATA_DIR, 'train-rle.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')

TEST_FILE = "../data/sample_submission.csv"

ID_COLUMN = 'ImageId'
ENCODING_COL = 'EncodedPixels'
