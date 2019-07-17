import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.transform import resize
import cv2

# from utils import get_crop_pad_sequence, run_length_decoding
import pdb
import config


def binarize(image, threshold):
    image_binarized = (image > threshold).astype(np.uint8)
    return image_binarized

