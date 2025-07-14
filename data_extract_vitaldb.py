import vitaldb
import numpy as np
import pandas as pd
import pickle as pkl
import os
import neurokit2 as nk
import multiprocessing as mp
from tqdm import tqdm

from data_load_preprocess.load_record import load_record
from data_load_preprocess.filters import zscore, butterworth
from data_load_preprocess.sample_from_record import sample_from_record
from data_load_preprocess.remove_samples import remove_samples
from data_load_preprocess.median_sample import compute_median_sample_ecg, compute_median_features_ecg, plot_median_ecg
from data_load_preprocess.feature_extraction.features_ecg import compute_ecg_morphology_features, ecg_features_relative_r

import config as cfg

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def process_case_id(case_id):
    """
    This function loads the record for a given case id, preprocesses the signals, extracts samples, and removes
    unsuitable samples based on physiological thresholds. The function also calculates the median waveforms and
    traditional features for the ECG signal.
    :param case_id:int        case id for which the samples are extracted
    :return:
           nr_of_samples:int  number of suitable samples for the current case id
    """
    # ======================================================================================================
    # 1. LOADING RECORDS
    # ======================================================================================================

    # Check if file for current case id exists and the samples have been loaded and preprocessed before
    file_path = os.path.join(cfg.path_data, 'case_ids', f'{case_id}.parquet')
    if os.path.exists(file_path):
        # Load the samples from the file
        df_samples = pd.read_parquet(file_path)
        print(f"Case ID {case_id}: {len(df_samples)} samples loaded.")
        return case_id, len(df_samples)

    # Load the record for the current case id if all the required tracks contain data
    record = load_record(case_id=case_id, track_names=cfg.track_names,
                         sampling_frequency=cfg.sampling_frequency)

    # ======================================================================================================
    # 2. FILTERING SIGNALS AND PREPROCESSING
    # ======================================================================================================

    # Remove leading and tailing nan values
    non_nan_rows = ~np.isnan(record).all(axis=1)
    start_index = np.argmax(non_nan_rows)  # First non-NaN row
    end_index = len(non_nan_rows) - np.argmax(non_nan_rows[::-1])  # Last non-NaN row + 1
    # Slice the array to remove leading and trailing NaN rows
    record = record[start_index:end_index]

    # ECG data_load_preprocess
    index_ecg = cfg.track_names.index("SNUADC/ECG_II")
    track_ecg = record[:, index_ecg]

    # Z-score normalization
    track_ecg_zscore = zscore(track_ecg)
    # Remove baseline wander using Highpass Butterworth zero-phase filtering
    if cfg.butter_hp_ecg:
        track_ecg_butter_hp = butterworth(data=track_ecg_zscore, btype='high', cutoff_frequency=cfg.butter_hp_ecg,
                                          sampling_frequency=cfg.sampling_frequency, filter_order=cfg.butter_order)
    else:
        track_ecg_butter_hp = track_ecg_zscore
    if cfg.butter_lp_ecg:
        # Remove high-frequency noise using Lowpass Butterworth zero-phase filtering
        track_ecg_butter_hplp = butterworth(data=track_ecg_butter_hp, btype='low', cutoff_frequency=cfg.butter_lp_ecg,
                                            sampling_frequency=cfg.sampling_frequency, filter_order=cfg.butter_order)
    else:
        track_ecg_butter_hplp = track_ecg_butter_hp

    record[:, index_ecg] = track_ecg_butter_hplp

    # ======================================================================================================
    # 3. EXTRACTING SAMPLES AND REMOVING UNSUITABLE SAMPLES
    # ======================================================================================================

    # Create samples by extracting a 20-second ABP fragment before each SV measurement
    samples_wave_tracks, indices_samples = sample_from_record(record=record, track_names=cfg.track_names,
        sampling_frequency=cfg.sampling_frequency, sample_length=cfg.sample_length)

    # Perform Z-score normalization
    if len(samples_wave_tracks) == 0:
        print(f"Case ID {case_id}: {len(samples_wave_tracks)} samples loaded.")
        return case_id, 0
    idx_ecg = cfg.track_names.index("SNUADC/ECG_II")
    samples_wave_tracks[:, :, idx_ecg] = np.apply_along_axis(zscore, 1, samples_wave_tracks[:, :, idx_ecg])

    # Remove samples that are unsuitable for training the NN model based on physiological thresholds
    df_samples = remove_samples(samples_wave_tracks=samples_wave_tracks, indices_samples=indices_samples,
        track_names=cfg.track_names, sampling_frequency=cfg.sampling_frequency, pvc_threshold=cfg.pvc_threshold)

    # ======================================================================================================
    # 4. CALCULATING MEDIAN WAVEFORMS AND TRADITIONAL FEATURES
    # ======================================================================================================

    # Extract median sample
    df_samples[['ecg_sample_median', 'ecg_samples_aligned']] = pd.DataFrame([
        compute_median_sample_ecg(sample, r_peaks, cfg.sampling_frequency, return_aligned=True)
        for sample, r_peaks in zip(df_samples['ecg_sample'], df_samples['ECG_R_Peaks'])])

    # Initialize the feature columns in df_samples
    feature_list = [
        "ECG_P_Peaks", "ECG_P_Onsets", "ECG_P_Offsets",
        "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_S_Peaks",
        "ECG_T_Peaks", "ECG_T_Onsets", "ECG_T_Offsets",
        "ECG_P_Duration", "ECG_Q_Duration", "ECG_R_Duration", "ECG_S_Duration", "ECG_T_Duration",
        "ECG_PR_Interval", "ECG_QRS_Complex", "ECG_ST_Interval", "ECG_ST_Segment", "ECG_QT_Interval",
        "ECG_P_Prominence", "ECG_Q_Prominence", "ECG_R_Prominence", "ECG_S_Prominence", "ECG_T_Prominence"
    ]
    features_dict = {feature: [] for feature in feature_list}
    valid_sample_mask = np.ones(len(df_samples), dtype=bool)

    if cfg.butter_lp_ecg and cfg.butter_hp_ecg:
        # Define all full length ecg samples and the lists of R-peak indices
        ecg_samples = df_samples['ecg_sample']
        r_peaks_lists = df_samples['ECG_R_Peaks']
        ecg_samples_median = df_samples['ecg_sample_median']
        ecg_samples_aligned = df_samples['ecg_samples_aligned']

        for idx_sample, (ecg_sample, r_peaks, ecg_sample_median, samples_aligned) in enumerate(zip(ecg_samples, r_peaks_lists, ecg_samples_median, ecg_samples_aligned)):

            # Extract the peaks, onsets, and offsets of the ECG signal using Neurokit2
            try:
                ecg_features = nk.ecg_delineate(ecg_sample, r_peaks, sampling_rate=cfg.sampling_frequency)[1]
            except:
                for feature in feature_list:
                    features_dict[feature].append(np.nan)
                continue

            # Check if all features have the same length as ECG_R_Peaks, if not, we cannot reliably calculate the
            # morphology features, skip the sample

            if not all(len(ecg_features[feature]) == len(r_peaks) for feature in ecg_features.keys()) or len(r_peaks) == 0:
                valid_sample_mask[idx_sample] = False
                for feature in feature_list:
                    features_dict[feature].append(np.nan)
                continue

            # Calculate the relative x (time) and y (amplitude) locations of key ECG features with respect to the R-peak
            ecg_features_relative = ecg_features_relative_r(ecg_sample, ecg_features, r_peaks, cfg.sampling_frequency)

            # Compute the clinically relevant morphology features of the ECG signal
            ecg_features_morphology = compute_ecg_morphology_features(ecg_features_relative, len(r_peaks))

            # Compute the median of all ECG features (neurokit2 and clinical features) over all beats
            ecg_features_median = compute_median_features_ecg(ecg_features_relative, ecg_features_morphology)

            # # Plot the median ECG waveform and overlay detected feature points
            # ecg_sample_median_norm = ecg_sample_median - np.median(ecg_sample[r_peaks])
            # samples_aligned_norm = samples_aligned - np.median(ecg_sample[r_peaks])
            # median_fiducials = {k: ecg_features_median[k] for k in feature_list[:10] + ['ECG_R_Peaks'] if k in ecg_features_median}
            # plot_median_ecg(ecg_sample_median_norm, cfg.sampling_frequency, ecg_features_median=median_fiducials,
            #                 ecg_features_relative=ecg_features_relative, ecg_samples_aligned=samples_aligned_norm)

            # Save the median ECG features
            for feature in feature_list:
                features_dict[feature].append(ecg_features_median[feature])

    else:
        # If no Butterworth filter is applied, we cannot reliably calculate the morphology features, skip the sample
        valid_sample_mask = np.ones(len(df_samples), dtype=bool)
        for feature in feature_list:
            features_dict[feature] = [np.nan] * len(df_samples)

    # Efficiently add results to the DataFrame
    for feature in feature_list:
        df_samples[feature] = features_dict[feature]
    # Remove samples that were skipped due to missing features
    df_samples = df_samples[valid_sample_mask]

    # Save all suitable samples to a .npy file
    df_samples.to_parquet(file_path, engine="pyarrow")
    print(f"Case ID {case_id}: {len(df_samples)} samples loaded.")
    return case_id, len(df_samples)


if __name__ == "__main__":

    # Create folders and subfolders if they don't exist yet
    if not os.path.exists(os.path.join(cfg.path_data, "case_ids")):
        os.makedirs(os.path.join(cfg.path_data, "case_ids"))  # makedirs creates more sublevels if needed.

    # Find all case id's in VitalDB that have the required tracks
    print(f"Loading {', '.join(cfg.track_names)}")
    all_case_ids = sorted(vitaldb.find_cases(cfg.track_names))

    if cfg.parallel_preprocessing:
        # Process the case id's in parallel using all available CPUs
        num_workers = mp.cpu_count()
        with mp.Pool(num_workers) as pool:
            # For each case ID: load the record, extract samples from the record, remove unsuitable samples
            results = list(tqdm(pool.imap(process_case_id, all_case_ids), total=len(all_case_ids)))
    else:
        # Process the case id's sequentially
        results = []
        with tqdm(total=len(all_case_ids)) as pbar:
            for cur_case_id in all_case_ids:
                results.append(process_case_id(cur_case_id))
                pbar.update(1)

    filtered_results = [r for r in results if r[1] != 0]
    case_ids, sample_counts = zip(*filtered_results)

    sample_counts = list(sample_counts)
    case_ids = list(case_ids)

    # Save case IDs and total number of samples
    total_samples = sum(sample_counts)
    data_stats = {'case_ids': case_ids, 'samples_per_case_id': sample_counts, 'samples_total': total_samples}

    with open(os.path.join(cfg.path_data, 'data_stats.pkl'), 'wb') as f:
        pkl.dump(data_stats, f)  # type: ignore
