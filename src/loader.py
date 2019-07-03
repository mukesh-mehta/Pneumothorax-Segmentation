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

def load_train_val_dataset(batch_size= 16, num_workers=6, dev_mode=False):
    dataframe = pd.read_csv(config.ENCODING_FILE)
    if dev_mode:
        dataframe = dataframe.iloc[:100]
    train, val = train_test_split_stratified(dataframe)

    train_set = LoadDataset(list(train[config.ID_COLUMN].values))
    val_set = LoadDataset(list(val[config.ID_COLUMN].values))

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_set, batch_size=int(batch_size/2), shuffle=False, num_workers=num_workers)
    train_loader.num = len(train_set)
    val_loader.num = len(val_set)
    return train_loader, val_loader


def get_test_loader(batch_size=16, index=0, dev_mode=False):
    test_meta = pd.read_csv(config.TEST_FILE)
    # test_meta = test_meta.drop_duplicates(config.ID_COLUMN, keep='last').reset_index(drop=True)
    test_meta = test_meta[config.ID_COLUMN].values
    if dev_mode:
        test_meta = test_meta[:10]
    test_set = LoadDataset(list(test_meta),is_test = True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6,drop_last=False)
    test_loader.num = len(test_set)
    test_loader.meta = test_meta
    return test_loader

if __name__ == '__main__':
    train_loader, val_loader = load_train_val_dataset(batch_size = 10, num_workers=6)
    for image, mask in train_loader:
        print("check train")
        print(image.shape, mask.shape)
        break
    for image, mask in val_loader:
        print("check val")
        print(image.shape, mask.shape)
        break
    print("checked...")