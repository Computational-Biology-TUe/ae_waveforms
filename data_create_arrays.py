import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg


# Load the data statistics
data_stats_file = os.path.join(cfg.path_data, 'data_stats.pkl')
data_stats = pd.read_pickle(data_stats_file)
case_ids = data_stats['case_ids']
nr_of_samples = data_stats['samples_total']

# Allocate memory to not truncate the data
# (data type needs to be float64, but too little memory that can be allocated)
section_duration = 0.8 * (60 / 75)  # seconds
section_length = int(section_duration * cfg.sampling_frequency)  # in samples
np_waves = np.ndarray((nr_of_samples, section_length), dtype=np.float32)
np_features = np.ndarray((nr_of_samples, 35))
np_info = np.ndarray((nr_of_samples, 2))

# Reformat all the data into the right array structures for the DL algorithm
idx = 0
with tqdm(total=len(case_ids)) as pbar:  # define progress bar
    pbar.set_description('Stacking case ID arrays of samples...')
    for case_id in case_ids:

        file_path = os.path.join(cfg.path_data, 'case_ids', f'{case_id}.parquet')
        df_samples = pd.read_parquet(file_path)
        curr_nr_of_samples = len(df_samples)

        try:
            np_nk = np.array(df_samples.iloc[:, -25:-15].map(lambda d: [d['x'], d['y']]).values.tolist()
                             ).reshape(len(df_samples), -1)
            np_traditional = np.array(df_samples.iloc[:, -15:])
        except TypeError:
            # If the data is empty, skip this case_id
            np_nk = np.zeros((len(df_samples), 20))
            np_traditional = np.zeros((len(df_samples), 15))
        np_c = np.repeat(np.array([[case_id]]), curr_nr_of_samples, axis=0).astype(int)
        np_i = np.stack(df_samples.indices_samples.values)

        np_waves[idx:idx + curr_nr_of_samples, :] = np.stack(df_samples.ecg_sample_median.values)
        np_features[idx:idx + curr_nr_of_samples, :] = np.concatenate([np_nk, np_traditional], axis=1)
        np_info[idx:idx + curr_nr_of_samples, :] = np.concatenate([np_c, np_i], axis=1)

        idx += curr_nr_of_samples
        pbar.update(1)

# Save each array to one file
print("Saving data to .npy files...")
save_path = cfg.path_data
if not os.path.exists(save_path):
    os.mkdir(save_path)
np.save(f"{save_path}np_waves", np_waves, allow_pickle=cfg.use_pickle)
np.save(f"{save_path}np_features", np_features, allow_pickle=cfg.use_pickle)
np.save(f"{save_path}np_info", np_info, allow_pickle=cfg.use_pickle)
