import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_datasets(dataset_dir):
    """
    Loads the datasets from the .npy files.
    :param dataset_dir:str    The directory where the .npy files are located
    :return:
           all_x:np.array     ECG wave data, single 0.64 second 500Hz segment. Shape (batch, 320)
           all_a:np.array     Extracted fiducial points and traditionally used ECG waveform features. Shape (batch, 35)
           all_c:np.array     Case ID's and time indices of each data point (used to split dataset on patients). Shape (batch, 2)
    """
    load_data = {}
    for file in ['waves', 'features', 'info']:
        filepath = os.path.join(dataset_dir, f"np_{file}.npy")

        # Load from the .npy files
        load_data[file] = np.load(filepath, mmap_mode='r+')

    return load_data['waves'], load_data['features'], load_data['info']