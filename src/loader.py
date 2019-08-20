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

from utils import load_image, train_test_split_stratified, create_file_list
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)



def augment(image, mask=None, test=False, is_clf=False):
    AUGMENTATIONS_TRAIN = Compose([
                                HorizontalFlip(p=0.5),
                                OneOf([
                                    RandomContrast(),
                                    RandomGamma(),
                                    RandomBrightness(),
                                     ], p=0.3),
                                OneOf([
                                    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                    ], p=0.3),
                                # RandomSizedCrop(min_max_height=(128, 256), height=h, width=w,p=0.5),
                                ToFloat(max_value=1)
                            ],p=1)

    AUGMENTATIONS_TEST = Compose([
                            ToFloat(max_value=1)
                        ],p=1)
    if test:
        augmented = AUGMENTATIONS_TEST(image=image)
        img = torch.from_numpy(image).float().permute([2, 0, 1])
        return img
    else:
        augmented = AUGMENTATIONS_TRAIN(image=image, mask=mask)
        img = torch.from_numpy(augmented['image']).float().permute([2, 0, 1])
        mask = torch.from_numpy(augmented['mask']).float()
        return img, mask

def augment_clf(image, label=None, test=False, is_clf=False):
    AUGMENTATIONS_TRAIN = Compose([
                                HorizontalFlip(p=0.5),
                                OneOf([
                                    RandomContrast(),
                                    RandomGamma(),
                                    RandomBrightness(),
                                     ], p=0.3),
                                OneOf([
                                    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                    GridDistortion(),
                                    OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                    ], p=0.3),
                                # RandomSizedCrop(min_max_height=(128, 256), height=h, width=w,p=0.5),
                                ToFloat(max_value=1)
                            ],p=1)

    AUGMENTATIONS_TEST = Compose([
                            ToFloat(max_value=1)
                        ],p=1)
    if test:
        augmented = AUGMENTATIONS_TEST(image=image)
        img = torch.from_numpy(image).float().permute([2, 0, 1])
        return img
    else:
        augmented = AUGMENTATIONS_TRAIN(image=image)
        img = torch.from_numpy(augmented['image']).float().permute([2, 0, 1])
        return img, label

class LoadDatasetCLF(data.Dataset):
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
            image_path = os.path.join(config.TEST_IMG_DIR, file_id[0] + ".png")
            image = load_image(image_path)
            return augment_clf(image, test=True)
        else:
            image_path = os.path.join(config.TRAIN_IMG_DIR, file_id[0] + ".png")
            image = load_image(image_path)
            return augment_clf(image, label = file_id[1])


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
            return augment(image, test=True)
        else:
            image_path = os.path.join(config.TRAIN_IMG_DIR, file_id + ".png")
            mask_path = os.path.join(config.TRAIN_MASK_DIR, file_id + ".png")

            image = load_image(image_path)
            mask = load_image(mask_path, mask = True)
            return augment(image, mask=mask)

def load_train_val_dataset(batch_size= 16, num_workers=8, dev_mode=False, is_clf=False):
    dataframe = pd.read_csv(config.ENCODING_FILE)
    dataframe.drop_duplicates(subset=[config.ID_COLUMN], inplace=True)
    dataframe.loc[dataframe[config.ENCODING_COL]!='-1', "has_pneumo"] = 1
    dataframe.loc[dataframe[config.ENCODING_COL]=='-1', "has_pneumo"] = 0
    dataframe = dataframe.sample(frac=1, random_state=2)
    if is_clf:
        dataframe = dataframe[[config.ID_COLUMN, "has_pneumo"]]
        if dev_mode:
            dataframe = dataframe.iloc[:100]
        train, val = train_test_split_stratified(dataframe)

        train_set = LoadDatasetCLF(create_file_list(train))
        val_set = LoadDatasetCLF(create_file_list(val))

        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=False)
        val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
        train_loader.num = len(train_set)
        val_loader.num = len(val_set)
        return train_loader, val_loader
    else:
        if dev_mode:
            dataframe = dataframe.iloc[:100]
        train, val = train_test_split_stratified(dataframe)

        train_set = LoadDataset(list(train[config.ID_COLUMN].values))
        val_set = LoadDataset(list(val[config.ID_COLUMN].values))

        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=False)
        val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
        train_loader.num = len(train_set)
        val_loader.num = len(val_set)
        return train_loader, val_loader


def get_test_loader(batch_size=16, index=0, dev_mode=False):
    test_meta = pd.read_csv(config.TEST_FILE)
    # test_meta = test_meta.drop_duplicates(config.ID_COLUMN, keep='last').reset_index(drop=True)
    test_meta = test_meta[config.ID_COLUMN].values
    if dev_mode:
        test_meta = test_meta[:25]
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
    test_loader = get_test_loader(batch_size=16, index=0, dev_mode=True)
    for image in test_loader:
        print("check test")
        print(image.shape)
    print("checked...")