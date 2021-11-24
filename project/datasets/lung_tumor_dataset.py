import os
import numpy as np
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch
from torch.utils.data import DataLoader


class LungTumorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, masks_dir, transform=None):
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.all_file_names = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.all_file_names)

    def augment(self, data, mask):
        # Fix to lack of randomness problem when using something other than pytorch
        random_seed = torch.randint(0, 1000000, (1,)).item()
        ia.seed(random_seed)

        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
        return aug_data, aug_mask.get_arr()

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.all_file_names[idx])
        mask_path = os.path.join(self.masks_dir, self.all_file_names[idx])
        data = np.load(data_path)
        mask = np.load(mask_path)
        if self.transform:
            data, mask = self.augment(data, mask)
        return np.moveaxis(data, -1, 0), np.expand_dims(mask, axis=0)


def get_datasets(preprocessed_input_dir, aug_pipeline):
    train_data_dir = os.path.join(preprocessed_input_dir, 'train', 'data')
    train_label_dir = os.path.join(preprocessed_input_dir, 'train', 'mask')
    validation_data_dir = os.path.join(preprocessed_input_dir, 'val', 'data')
    validation_label_dir = os.path.join(preprocessed_input_dir, 'val', 'data')

    training_data = LungTumorDataset(train_data_dir, train_label_dir, transform=aug_pipeline)
    validation_data = LungTumorDataset(validation_data_dir, validation_label_dir, transform=None)
    return training_data, validation_data

