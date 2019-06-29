"""
TODO: 
1. Augmentations
2. TTA
"""
import os
import config
import pandas as pd
import torch
from torch.utils import data

from utils import load_image, train_test_split_stratified

class LoadDataset(data.Dataset):
    def __init__(self, file_list, is_test = False):
        self.is_test = is_test
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        file_id = self.file_list[index]
        
        if self.is_test:
            image_path = os.path.join(config.TEST_IMG_DIR, file_id + ".png")
            image = load_image(image_path)
            return (image,)
        else:
            image_path = os.path.join(config.TRAIN_IMG_DIR, file_id + ".png")
            mask_path = os.path.join(config.TRAIN_MASK_DIR, file_id + ".png")

            image = load_image(image_path)
            mask = load_image(mask_path, mask = True)
            return image, mask

def load_train_val_dataset():
    dataframe = pd.read_csv(config.ENCODING_FILE)
    train, val = train_test_split_stratified(dataframe)
    return LoadDataset(list(train[config.ID_COLUMN].values)), LoadDataset(list(val[config.ID_COLUMN].values))

if __name__ == '__main__':
    train, val = load_train_val_dataset()
    for image, mask in data.DataLoader(train, batch_size = 10,shuffle=True, num_workers=6):
        print("check train")
        print(image.shape, mask.shape)
        break
    for image, mask in data.DataLoader(val, batch_size = 10,shuffle=True, num_workers=6):
        print("check val")
        print(image.shape, mask.shape)
        break
    print("checked...")