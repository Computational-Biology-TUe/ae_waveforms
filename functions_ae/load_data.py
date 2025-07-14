import os
import re
import time
import glob
import logging

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset  # data loading


class Datasets(Dataset):
    """Characterizes a dataset for PyTorch - torch.utils.data"""

    def __init__(self, all_x, all_a, all_c, flag=None, swap_dim=True):
        """ Initialization of the Datasets object. Values true in flag will be selected from the arrays"""
        self.flag = flag

        all_y = all_x
        swap_y = True

        if self.flag is None:
            self._wave = all_x
            self._aswh = all_a
            self._ylabel = all_y
            self._chart = all_c
        else:
            self._wave = all_x[self.flag]
            self._aswh = all_a[self.flag]
            self._ylabel = all_y[self.flag]
            self._chart = all_c[self.flag]

        if swap_dim:
            self._wave = self._wave[:, np.newaxis, :]
        else:
            self._wave = self._wave[:, :, np.newaxis]
        self.wave_pt = torch.tensor(self._wave, dtype=torch.float32)
        self.aswh_pt = torch.tensor(self._aswh, dtype=torch.float32)
        if swap_y:
            if swap_dim:
                self._ylabel = self._ylabel[:, np.newaxis, :]
            else:
                self._ylabel = self._ylabel[:, :, np.newaxis]
            self.ylabel_pt = torch.tensor(self._ylabel, dtype=torch.float32)
        else:
            self.ylabel_pt = torch.tensor(self._ylabel, dtype=torch.float32).unsqueeze(1)
        self.chart_pt = self._chart

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.chart_pt)

    def __getitem__(self, index):
        """Get specific index from dataset"""
        x = self.wave_pt[index]
        f = self.aswh_pt[index]
        y = self.ylabel_pt[index]
        c = self.chart_pt[index]

        return x, y, f, c

    def get_yvals(self):
        return self.ylabel_pt

    def get_patient_demographics(self):
        demographics = np.concatenate([self.chart_pt.reshape(len(self.chart_pt), 1), self.aswh_pt], axis=1)
        return pd.DataFrame(demographics, columns=['id', 'age', 'sex', 'weight', 'height']
                            ).groupby('id').mean().reset_index()


def load_datasets(dataset_dir):
    """
    Loads the datasets from the .npy files.
    :param dataset_dir:str    The directory where the .npy files are located
    :return:
           all_x:np.array     ECG wave data, single 0.64 second 500Hz segment. Shape (batch, 320)
           all_a:np.array     Extracted fiducial points and traditionally used ECG waveform features. Shape (batch, 35)
           all_c:np.array     Case ID's and time indices of each data point (used to split dataset on patients). Shape (batch, 2)
    """
    logger = logging.getLogger('load_datasets')

    logger.info(f"\tLoad data")

    load_data = {}
    for file in ['waves', 'features', 'info']:
        filepath = os.path.join(dataset_dir, f"np_{file}.npy")

        # Load from the .npy files
        logger.info(f"\tLoading data from {filepath}")
        load_data[file] = np.load(filepath, mmap_mode='r+')

    return load_data['waves'], load_data['features'], load_data['info']


def split_datasets(data_c, dataset_dir, results_dir=None, sampling_rate_val=0.2, sampling_rate_test=0.1,
                   regex_rule=None, nr_train_ids=None):
    """
    Split the dataset into train, validation and test sets based on the unique ids.
    :param data_c:np.ndarray            Chart data, case ID's of each data point used to split the dataset.
    :param dataset_dir:str              Directory where the .npy files are located
    :param results_dir:str              Directory where the indices for the splits and the trained network will be saved
    :param sampling_rate_val:float      Percentage of the dataset to be used for validation
    :param sampling_rate_test:float     Percentage of the dataset to be used for testing
    :param regex_rule:NoneType          Regex rule to be used for filtering the dataset
    :param nr_train_ids:int             Number of training case IDs to use, set to 0 to use all available IDs
    :return:
           indices_train:np.ndarray     Indices of the training set
           indices_val:np.ndarray       Indices of the validation set
           indices_test:np.ndarray      Indices of the test set
    """
    file_train = os.path.join(dataset_dir, 'indices_train_*.npy')
    file_val = os.path.join(dataset_dir, 'indices_val_*.npy')
    file_test = os.path.join(dataset_dir, 'indices_test_*.npy')
    if glob.glob(file_train) and glob.glob(file_val) and glob.glob(file_test):
        indices_train = np.load(glob.glob(file_train)[0])
        indices_val = np.load(glob.glob(file_val)[0])
        indices_test = np.load(glob.glob(file_test)[0])

        if nr_train_ids > 0:
            # Limit the number of training IDs if specified
            unique_ids = np.unique(data_c[:, 0])
            if len(unique_ids) > nr_train_ids:
                selected_ids = np.random.choice(unique_ids, size=nr_train_ids, replace=False)
                indices_train = indices_train & np.isin(data_c[:, 0], selected_ids)

        # Save the indices for the splits to the results directory
        np.save(os.path.join(results_dir, os.path.basename(glob.glob(file_train)[0])), indices_train)
        np.save(os.path.join(results_dir, os.path.basename(glob.glob(file_val)[0])), indices_val)
        np.save(os.path.join(results_dir, os.path.basename(glob.glob(file_test)[0])), indices_test)

    else:
        # Calculate sampling rate for the train split
        sampling_rate_train = round(1 - sampling_rate_val - sampling_rate_test, 2)

        unique_ids = np.unique(data_c[:,0])
        random_index = np.random.choice(['train', 'val', 'test'], size=len(unique_ids),
                                        p=[sampling_rate_train, sampling_rate_val, sampling_rate_test])

        train_chart = unique_ids[np.where(random_index == 'train')]
        val_chart = unique_ids[np.where(random_index == 'val')]
        test_chart = unique_ids[np.where(random_index == 'test')]

        sel_idx_train = np.array([c in train_chart for c in data_c[:,0]])
        sel_idx_val = np.array([c in val_chart for c in data_c[:,0]])
        sel_idx_test = np.array([c in test_chart for c in data_c[:,0]])

        if regex_rule is not None:
            r1 = re.compile(regex_rule)
            v_match_1 = np.vectorize(lambda x: bool(r1.match(x)))
            regex_matched_index = np.array(v_match_1(data_c[:,0]))
        else:
            regex_matched_index = np.array([True] * len(data_c[:,0]))

        indices_train = (sel_idx_train & regex_matched_index)
        indices_val = (sel_idx_val & regex_matched_index)
        indices_test = (sel_idx_test & regex_matched_index)

        # Save the indices for the splits to the dataset directory
        np.save(file_train.replace('*', time.strftime('%y%m%d')), indices_train)
        np.save(file_val.replace('*', time.strftime('%y%m%d')), indices_val)
        np.save(file_test.replace('*', time.strftime('%y%m%d')), indices_test)

        if nr_train_ids > 0:
            # Limit the number of training IDs if specified
            unique_ids = np.unique(data_c[:, 0])
            if len(unique_ids) > nr_train_ids:
                selected_ids = np.random.choice(unique_ids, size=nr_train_ids, replace=False)
                indices_train = indices_train & np.isin(data_c[:, 0], selected_ids)

        # Save the indices for the splits to the network training results directory
        np.save(os.path.join(results_dir, f"indices_train_{time.strftime('%y%m%d')}"), indices_train)
        np.save(os.path.join(results_dir, f"indices_val_{time.strftime('%y%m%d')}"), indices_val)
        np.save(os.path.join(results_dir, f"indices_test_{time.strftime('%y%m%d')}"), indices_test)

    # Save the subject IDs used for the splits to the network training results directory
    save_subject_ids(data_c[:,0][indices_train], 'training', results_dir)
    save_subject_ids(data_c[:,0][indices_val], 'validation', results_dir)
    save_subject_ids(data_c[:,0][indices_test], 'testing', results_dir)

    return indices_train, indices_val, indices_test


def save_subject_ids(subject_ids, dataset, results_dir):
    """
    Record used subject IDs for given dataset in a .csv file
    :param subject_ids:np.ndarray   Subject id's belonging to the samples that ended up in 'dataset'
    :param dataset:str              Name of the dataset
    :param results_dir:str          Directory where the results will be saved
    """
    subject_ids_unique, counts_per_id = np.unique(subject_ids, return_counts=True)
    with open(os.path.join(results_dir, f"subject_ids_used_in_{dataset}.csv"), 'wt') as f:
        f.write('chart_names, counts\n')
        for o1, o2 in zip(subject_ids_unique, counts_per_id):
            f.write(f"{o1}, {o2}\n")
