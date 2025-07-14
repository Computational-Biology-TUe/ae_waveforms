import numpy as np
import pandas as pd
from scipy.stats import entropy
import neurokit2 as nk
import matplotlib.pyplot as plt


def remove_samples(samples_wave_tracks, indices_samples, track_names, sampling_frequency=500, pvc_threshold=150,
                   sd_threshold_ecg=0.05):
    """
    Applies beat-detection (R-peak) and removes samples with an HR <30 bpm or >180 bpm, with frequent (>50%) premature
    ventricular contractions, or with a large SD compared to nr of found beats ratio > 0.0335.
    :param samples_wave_tracks:np.ndarray   20 second waveform segments
    :param indices_samples:np.ndarray       time indices of the samples at the end of the waveform samples
    :param track_names:list                 track names
    :param sampling_frequency:int           sampling frequency of the data in Hz
    :param pvc_threshold:float              threshold for the RMSSD of the HRV to determine if there are PVCs
    :param sd_threshold_ecg:int             threshold for the standard deviation of the ECG
    :return:
           samples_wave_tracks:np.ndarray   20 second waveform segments without the removed samples
    """
    indices_to_be_removed = set()

    # Map the wave tracks to their respective column names
    wave_map = {'SNUADC/ECG_II': 'ecg_sample'}
    # Find the available wave tracks
    wave_track_names = [item for item in track_names if item in wave_map]

    # 1. NAN FILTERING
    # Remove any sample that contains NaN values
    if samples_wave_tracks is not None:
        for idx in range(samples_wave_tracks.shape[2]):
            indices_to_be_removed.update(np.where(np.any(np.isnan(samples_wave_tracks[:, :, idx]), axis=1))[0])

    # 2. WAVE SAMPLE FILTERING
    # Add available tracks and the indices_samples to df_samples
    df_samples = pd.DataFrame({
        **{wave_map[name]: list(samples_wave_tracks[:, :, wave_track_names.index(name)]) for name in wave_track_names},
        'indices_samples': list(indices_samples)
    })

    # Filter the DataFrame to remove unsuitable samples
    df_samples = df_samples.drop(index=list(indices_to_be_removed)).reset_index(drop=True)
    indices_to_be_removed.clear()

    # 3 ECG FILTERING
    if len(df_samples):
        # Detect peaks in the ECG signal and compute HR
        # pbar.set_description(f"Processing case_id={case_id}: Detecting ECG peaks")
        df_samples['ECG_R_Peaks'] = df_samples['ecg_sample'].apply(
            lambda x: list(nk.ecg_findpeaks(x, sampling_rate=sampling_frequency)["ECG_R_Peaks"]))
        # Remove R-peaks out of bounds
        df_samples['ECG_R_Peaks'] = df_samples['ECG_R_Peaks'].apply(
            lambda peaks: [peak for peak in peaks if 0 < peak < len(df_samples['ecg_sample'].iloc[0])])
        df_samples['ecg_hr'] = df_samples['ECG_R_Peaks'].apply(
            lambda peaks: estimate_hr([], peaks, sampling_frequency, mod='max'))

        # Compute the Root Mean Square of Successive Differences (RMSSD) as a measure of HRV
        df_samples['ecg_hrv_rmssd'] = df_samples['ECG_R_Peaks'].apply(lambda peaks: rmssd(peaks, sampling_frequency))

        # Calculate SD of the ECG signal
        df_samples['ecg_sd'] = df_samples['ecg_sample'].apply(lambda x: np.std(x))
        # Compute the SD / median HR ratio
        df_samples['ecg_sd/hr'] = df_samples['ecg_sd'] / df_samples['ECG_R_Peaks'].apply(len)

        # Remove samples with:
        #   - HR <30 bpm or >180 bpm
        #   - Frequent (>50%) premature ventricular contractions
        #   - Large SD compared to nr of found beats ratio > 0.0335
        indices_to_be_removed = set(df_samples[
                                        (df_samples['ecg_hr'] < 30) | (df_samples['ecg_hr'] > 180) |
                                        (df_samples['ecg_hr'].isna()) |
                                        (df_samples['ecg_hrv_rmssd'] > pvc_threshold) |
                                        (df_samples['ecg_sd/hr'] > sd_threshold_ecg)].index)

        # plot_error(df_samples)

        # Filter the DataFrame to remove unsuitable samples
        df_samples = df_samples.drop(index=list(indices_to_be_removed)).reset_index(drop=True)
        indices_to_be_removed.clear()

    return df_samples


def estimate_hr(min_peaks, max_peaks, sampling_frequency, mod='mean'):
    """
    Compute heart rate from the detected peaks.
    :param min_peaks:np.ndarray     locations of the min peaks
    :param max_peaks:np.ndarray     locations of the max peaks
    :param sampling_frequency:int   sampling frequency in Hz
    :param mod:str                  mode to compute hr: 'min', 'mean', 'max'
    :return:
            float                   hr in beats per minute (bpm)
    """
    min_intervals = np.diff(min_peaks)
    max_intervals = np.diff(max_peaks)

    if mod == 'min':  # compute hr based on min peaks intervals
        median_beat_sec = np.median(min_intervals)
    elif mod == 'mean':  # compute hr based on median of max and min peaks intervals
        median_beat_sec = np.median(np.concatenate([min_intervals, max_intervals]))
    else:  # compute hr based on max peaks interval
        if not mod == 'max':
            print('mod selection got wrong parameter: {}, Use "max" as default settings'.format(mod))
        median_beat_sec = np.median(max_intervals)

    return 60 * sampling_frequency / median_beat_sec


def rmssd(peaks, sampling_frequency=500):
    """
    Computes the Root Mean Square of Successive Differences (RMSSD) as a measure of HRV.
    Used to filter out Ventricular Premature Beats (PVCs).
    :param peaks:np.ndarray          indices of the peaks
    :param sampling_frequency:int    sampling frequency of the data in Hz
    :return:
           float                     the RMSSD
    """
    intervals = np.diff(peaks) / (sampling_frequency * 1000)  # in milliseconds
    successive_differences = np.diff(intervals)
    return np.sqrt(np.mean(successive_differences ** 2))


def sd_filter(samples, threshold=39):
    """
    Filters samples based on the standard deviation of each sample and a given threshold.
    :param samples:np.ndarray   20 second waveform segments
    :param threshold:int        threshold for the standard deviation
    :return:
            set                 indices of the unsuitable samples that are above the threshold
    """
    return set(np.where(np.std(samples, axis=1) > threshold)[0])


def calculate_entropy(signal, num_bins=100):
    """Calculate Shannon entropy of a signal by discretizing it into bins."""
    # Discretize signal into bins
    hist, _ = np.histogram(signal, bins=num_bins, density=True)

    # Normalize to get probability distribution
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)  # Compute entropy in bits


def autocorrelation(signal_data):
    """
    Calculate the autocorrelation of a signal.

    :param signal_data: The ECG signal (or any time-series signal)
    :return: The normalized autocorrelation
    """
    # Length of the signal
    n = len(signal_data)

    # Calculate autocorrelation
    autocorr = np.correlate(signal_data, signal_data, mode='full')

    # Normalize the autocorrelation so the peak at lag 0 is 1
    autocorr = autocorr / autocorr.max()

    # Only keep the second half (positive lags)
    autocorr = autocorr[n - 1:]

    return autocorr


def plot_error(df_samples, metric='ecg_sd/hr', sort_by='largest'):
    if sort_by == 'largest':
        # Get indices of the top 28 samples based on the metric
        top_indices = df_samples.nlargest(28, metric).index
    elif sort_by == 'evenly':
        # Sort the dataframe by the metric
        df_sorted = df_samples.sort_values(by=metric, ascending=True)

        # Select 28 evenly spaced indices
        top_indices = df_sorted.iloc[np.linspace(0, len(df_sorted) - 1, 28, dtype=int)].index
    else:
        raise ValueError("sort_by must be either 'largest' or 'evenly'")

    # Create a figure with 7 rows and 4 columns (28 subplots)
    fig, axes = plt.subplots(7, 4, figsize=(25, 13))
    axes = axes.flatten()

    # Plot the selected samples
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        ecg_sample = df_samples.loc[idx, 'ecg_sample']
        r_peaks = df_samples.loc[idx, 'ECG_R_Peaks']
        ratio_value = df_samples.loc[idx, metric]

        # Plot the ECG sample
        ax.plot(ecg_sample, label=f"ECG", color='black')
        # Mark R-peaks in green
        ax.scatter(r_peaks, ecg_sample[r_peaks], color='green', marker='o', label='R-peaks')

        ax.set_title(f"{metric}: {ratio_value:.4f}")
        ax.legend()
        ax.grid()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
