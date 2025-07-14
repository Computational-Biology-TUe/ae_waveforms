import numpy as np
import matplotlib.pyplot as plt


def plot_ecg_features(ecg_sample, ecg_features, sampling_frequency=500):
    """
    Plot the original ECG waveform with the 11 key features detected by NeuroKit.
    :param ecg_sample:np.ndarray        The ECG signal.
    :param ecg_features:dict            Containing the 11 feature indices from NeuroKit.
    :param sampling_frequency:int       Sampling frequency of the ECG signal (Hz).
    """
    # Generate time axis (in seconds)
    time_axis = np.linspace(0, len(ecg_sample) / sampling_frequency, len(ecg_sample))

    # Create figure
    plt.figure(figsize=(200, 6))
    plt.plot(time_axis, ecg_sample, label="ECG Signal", color="black", linewidth=1)

    # Define feature markers and colors
    feature_colors = {
        "ECG_P_Peaks": "blue",
        "ECG_P_Onsets": "lightblue",
        "ECG_P_Offsets": "deepskyblue",
        "ECG_Q_Peaks": "red",
        "ECG_R_Peaks": "green",
        "ECG_R_Onsets": "lightgreen",
        "ECG_R_Offsets": "darkgreen",
        "ECG_S_Peaks": "purple",
        "ECG_T_Peaks": "orange",
        "ECG_T_Onsets": "gold",
        "ECG_T_Offsets": "darkorange",
    }

    # Plot each feature with different markers
    for feature, color in feature_colors.items():
        if feature in ecg_features and ecg_features[feature] is not None:
            indices = ecg_features[feature]
            indices = np.array(indices)
            valid_indices = indices[~np.isnan(indices)].astype(int)  # Remove NaNs
            plt.scatter(time_axis[valid_indices], ecg_sample[valid_indices],
                        label=feature.replace("ECG_", "").replace("_", " "),
                        color=color, marker="o", s=40, edgecolors="black")

    # Labels and Legend
    plt.xlabel("Time (s)")
    plt.ylabel("ECG Amplitude")
    plt.title("ECG Waveform with Feature Annotations")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()

    # Show plot
    plt.tight_layout()
    plt.show()


def ecg_features_relative_r(ecg_sample, ecg_features, r_peaks, sampling_frequency):
    """
    Compute the relative x (time) and y (amplitude) locations of key ECG features with respect to the R-peak.
    :param ecg_sample:np.ndarray        The ECG waveform.
    :param ecg_features:dict            Containing ECG wave feature indices (from Neurokit).
    :param r_peaks:list                 Indices of detected R-peaks.
    :param sampling_frequency:int       Sampling frequency of the ECG signal in Hz.
    :return:
           ecg_features_relative:dict   Relative x and y locations for each the neurokit features across all beats.
    """

    ecg_features_relative = {}
    for feat in ecg_features.keys():

        ecg_features_relative[feat] = {"x": [], "y": []}

        # Compute relative positions for each detected beat
        for beat_nr, r_idx in enumerate(r_peaks):
            idx = ecg_features[feat][beat_nr]
            if idx is np.nan:
                relative_x = relative_y = np.nan
            else:
                # Relative x to R-peak (time in milliseconds)
                relative_x = (idx - r_idx) / sampling_frequency * 1000
                # Relative y to R-peak (amplitude difference)
                relative_y = ecg_sample[idx] - ecg_sample[r_idx]

            ecg_features_relative[feat]["x"].append(relative_x)
            ecg_features_relative[feat]["y"].append(relative_y)

    return ecg_features_relative


def compute_ecg_morphology_features(ecg_features, nr_of_beats):
    """
    Compute clinically relevant ECG morphological features from detected wave indices.
    :param ecg_features:dict            Indices of ECG wave components, as provided by Neurokit
                                        (e.g., 'ECG_P_Onsets', 'ECG_R_Peaks', etc.).
    :param nr_of_beats:int              Number of detected beats in the ECG signal.
    :return: 
           dict                         Computed ECG morphological features.
    """
    def time_diff(feature1, feature2):
        """Compute time difference (in seconds) between two ECG feature indices."""
        return np.array(ecg_features[feature2]['x']) - np.array(ecg_features[feature1]['x'])

    def amplitude_diff(peak_feature, baseline_feature):
        """Compute amplitude difference between a peak and its baseline."""
        if type(baseline_feature) is list:
            baseline_values = np.nanmax(np.stack([ecg_features[baseline_feature[0]]['y'],
                                                  ecg_features[baseline_feature[1]]['y']]), axis=0)
        else:
            baseline_values = ecg_features[baseline_feature]['y']
        return np.array(ecg_features[peak_feature]['y']) - np.array(baseline_values)

    ecg_features['ECG_R_Peaks'] = {'x': [0.0] * nr_of_beats, 'y': [0.0] * nr_of_beats}

    # Time-based features (in milliseconds)
    features = {
        "ECG_P_Duration": time_diff("ECG_P_Onsets", "ECG_P_Offsets"),
        "ECG_Q_Duration": time_diff("ECG_R_Onsets", "ECG_R_Peaks"),
        "ECG_R_Duration": time_diff("ECG_Q_Peaks", "ECG_S_Peaks"),
        "ECG_S_Duration": time_diff("ECG_R_Peaks", "ECG_R_Offsets"),
        "ECG_T_Duration": time_diff("ECG_T_Onsets", "ECG_T_Offsets"),
        "ECG_PR_Interval": time_diff("ECG_P_Onsets", "ECG_R_Peaks"),
        "ECG_QRS_Complex": time_diff("ECG_R_Onsets", "ECG_R_Offsets"),
        "ECG_ST_Interval": time_diff("ECG_R_Offsets", "ECG_T_Offsets"),
        "ECG_ST_Segment": time_diff("ECG_R_Offsets", "ECG_T_Onsets"),
        "ECG_QT_Interval": time_diff("ECG_R_Onsets", "ECG_T_Offsets"),
    }

    # Amplitude-based features (in Z-score normalized units)
    features.update({
        "ECG_P_Prominence": amplitude_diff("ECG_P_Peaks", ["ECG_P_Onsets", "ECG_P_Offsets"]),
        "ECG_Q_Prominence": amplitude_diff("ECG_Q_Peaks", "ECG_R_Onsets"),
        "ECG_R_Prominence": amplitude_diff("ECG_R_Peaks", ["ECG_Q_Peaks", "ECG_S_Peaks"]),
        "ECG_S_Prominence": amplitude_diff("ECG_S_Peaks", "ECG_R_Offsets"),
        "ECG_T_Prominence": amplitude_diff("ECG_T_Peaks", ["ECG_T_Onsets", "ECG_T_Offsets"]),
    })

    return features
