import os

import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset


class TagImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]['file_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        cat_name = self.data_frame.iloc[idx]['answer']
        sample['label'] = cat_name
        sample['image_name'] = img_name
        return sample


class TagImageInferenceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        sample['image_name'] = img_name
        return sample