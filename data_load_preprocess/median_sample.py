import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt


def compute_median_sample_ecg(sample_ecg, r_peaks, sampling_frequency, resample_ecg=False, return_aligned=False):
    """
    Compute the median ECG waveform by slicing around R-peaks and taking the median.

    :param sample_ecg:np.ndarray        The ECG signal.
    :param r_peaks:np.ndarray           Indices of detected R-peaks.
    :param sampling_frequency:int       Sampling frequency of the ECG signal.
    :param resample_ecg:bool            Whether to resample the median waveform to 30 Hz.
    :param return_aligned:bool          Whether to return the aligned sections as well.
    :return:
           np.ndarray                   The median ECG waveform.
    """
    # Define section duration based on capturing a slightly smaller section from one beat with normal heart rate of 75:
    # 0.8 * (60 / 75) seconds
    section_duration = 0.8 * (60 / 75)  # seconds
    section_length = int(section_duration * sampling_frequency)  # in samples

    # Define pre- and post-R-peak lengths (1/3 before, 2/3 after)
    pre_r_length = int(1/3 * section_length)
    post_r_length = section_length - pre_r_length

    # List to store extracted sections
    aligned_sections = []

    for r_peak in r_peaks:
        start_idx = max(0, r_peak - pre_r_length)
        end_idx = min(len(sample_ecg), r_peak + post_r_length)

        # Extract section
        section = sample_ecg[start_idx:end_idx]

        # Pad if section is shorter (e.g., near start or end of signal)
        if len(section) < section_length:
            padded_section = np.full(section_length, np.nan)
            padded_section[:len(section)] = section
            section = padded_section

        aligned_sections.append(section)

    # Convert to array and compute median per point (ignoring NaNs)
    aligned_sections = np.array(aligned_sections)
    median_waveform = np.nanmedian(aligned_sections, axis=0)

    if resample_ecg:
        median_waveform = resample(median_waveform, int(60 / 30 * sampling_frequency))

    if return_aligned:
        return median_waveform, aligned_sections
    return median_waveform


def compute_median_features_ecg(ecg_features, ecg_features_morphology):
    """
    Compute the relative x (time) and y (amplitude) locations of key ECG features 
    with respect to the R-peak, then take the median over all beats.
    :param ecg_features:dict                Containing ECG wave feature indices (from Neurokit).
    :param ecg_features_morphology:dict     Containing ECG morphology features.
    :return:
           ecg_features_relative:dict       Relative x and y locations for each the neurokit features across all beats.
           ecg_features_median:dict         Median x and y locations for each feature for the neurokit features,
                                            median values for the morphological features.

    """
    ecg_features_median = {}

    # 1. Compute the relative positions of each feature with respect to the R-peak and then compute the median over all
    # beats for each of the neurokit2-derived ECG features.
    for feat in ecg_features.keys():
        # Compute median relative positions
        ecg_features_median[feat] = {"x": np.nanmedian(ecg_features[feat]["x"]),
                                     "y": np.nanmedian(ecg_features[feat]["y"])}

    # 2. Compute the median over the additional ECG morphology features
    for feat in ecg_features_morphology.keys():
        ecg_features_median[feat] = np.nanmedian(ecg_features_morphology[feat])

    return ecg_features_median


def plot_median_ecg(ecg_sample_median, sampling_frequency, ecg_features_median=None, ecg_features_relative=None, ecg_samples_aligned=None):
    """
    Plot the median ECG waveform and overlay detected feature points.

    :param ecg_sample_median:np.ndarray     The median ECG waveform.
    :param sampling_frequency:int           Sampling frequency of the ECG signal in Hz.
    :param ecg_features_median:dict         Median relative feature points.
    :param ecg_features_relative:dict       All relative detected feature points across beats, x and y coordinates.
    :param ecg_samples_aligned:np.ndarray   Optional. Aligned ECG samples for visual comparison.
    """
    time_axis = (np.arange(len(ecg_sample_median)) / sampling_frequency - 106/sampling_frequency) * 1000

    plt.figure(figsize=(18, 6))

    # Plot all detected features (small + markers)
    if ecg_features_relative is not None:
        for feat, values in ecg_features_relative.items():
            plt.scatter(values["x"], values["y"], label=f"{feat} (all)", marker="+")

    # Plot median feature positions (o markers)
    if ecg_features_median is not None:
        for feat, values in ecg_features_median.items():
            plt.scatter(values["x"], values["y"], label=f"{feat} (median)", marker="o")

    if ecg_samples_aligned is not None:
        for i, sample in enumerate(ecg_samples_aligned):
            plt.plot(time_axis, sample, label=f"ECG Beat {i}", color='black', alpha=0.1)

    plt.plot(time_axis, ecg_sample_median, label="Median ECG", color='black')


    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV or signal units)")
    plt.title("ECG Median Waveform with Feature Points")
    plt.grid()

    plt.tight_layout()
    plt.show()
