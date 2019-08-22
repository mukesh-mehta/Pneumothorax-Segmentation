import os

HEIGHT = 512
WIDTH = 512

DATA_DIR = "../data/siim-png-images"

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR_RAW = os.path.join(TRAIN_DIR, 'train_png')
TRAIN_MASK_DIR_RAW =  os.path.join(TRAIN_DIR, 'train_masks')
TEST_IMG_DIR_RAW = os.path.join(TEST_DIR, 'test_png')

if HEIGHT!=1024:
	TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'train_png{}'.format(HEIGHT))
	TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'train_masks{}'.format(HEIGHT))
	TEST_IMG_DIR = os.path.join(TEST_DIR, 'test_png{}'.format(HEIGHT))
else:
	TRAIN_IMG_DIR =TRAIN_IMG_DIR_RAW 
	TRAIN_MASK_DIR = TRAIN_MASK_DIR_RAW
	TEST_IMG_DIR = TEST_IMG_DIR_RAW

ENCODING_FILE = os.path.join(DATA_DIR, 'train-rle.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')
CLF_MODEL_DIR = os.path.join(DATA_DIR, 'models_clf')

TEST_FILE = "../data/sample_submission.csv"

ID_COLUMN = 'ImageId'
ENCODING_COL = 'EncodedPixels'
