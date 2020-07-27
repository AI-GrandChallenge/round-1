import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_audio(path):
    sound = np.fromfile(path, dtype='h') / 32768.0
    return torch.Tensor(sound).unsqueeze(0)


def two_hot_encode(y1, y2, n_dim):
    ret = torch.zeros(y1.shape[0], n_dim)
    for idx in range(len(y1)):
        ret[idx, y1[idx]] = 1
        ret[idx, y2[idx]] = 1
    return ret


class AudioDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, root_dir: str):
        self.data_frame = data_frame
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.data_frame.iloc[idx]['file_name']+'.wav'
        audio_path = os.path.join(self.root_dir, audio_name)
        audio = load_audio(audio_path)

        sample['audio'] = audio
        start = self.data_frame.iloc[idx]['start']
        end = self.data_frame.iloc[idx]['end']
        sample['label'] = [start, end]
        sample['file_name'] = audio_name
        return sample


class AudioInferenceDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data_list = [img for img in os.listdir(self.root_dir) if not img.startswith('.')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.data_list[idx]
        audio_path = os.path.join(self.root_dir, audio_name)
        audio = load_audio(audio_path)
        sample['audio'] = audio
        sample['file_name'] = audio_name
        return sample