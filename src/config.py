import os

HEIGHT = 256
WIDTH = 256

DATA_DIR = "../data/siim-png-images"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR_RAW = os.path.join(TRAIN_DIR, 'train_png')
TRAIN_MASK_DIR_RAW =  os.path.join(TRAIN_DIR, 'train_masks')
TEST_IMG_DIR_RAW = os.path.join(TEST_DIR, 'test_png')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'train_png{}'.format(HEIGHT))
TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'train_masks{}'.format(HEIGHT))
TEST_IMG_DIR = os.path.join(TEST_DIR, 'test_png{}'.format(HEIGHT))

ENCODING_FILE = os.path.join(DATA_DIR, 'train-rle.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')
CLF_MODEL_DIR = os.path.join(DATA_DIR, 'models_clf')

TEST_FILE = "../data/sample_submission.csv"

ID_COLUMN = 'ImageId'
ENCODING_COL = 'EncodedPixels'
