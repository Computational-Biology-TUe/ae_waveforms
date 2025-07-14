import numpy as np


def sample_from_record(record, track_names, sampling_frequency=500, sample_length=20):
    """
    Sample 20-second segments from the full waveform record (ECG, ABP, and/or CVP) by slicing without overlap.
    :param record:np.ndarray                the full record with waveform (ABP, ECG, CVD) with shape:
                                            (nr_of_timepoints, nr_of_tracks)
    :param track_names:list                 string track names available in the record
    :param sampling_frequency:int           sampling frequency of the data in Hz
    :param sample_length:int                length of the DL input samples in seconds
    :return:
            samples_wave_tracks:np.ndarray  20 second wave segments with shape:
                                            (nr_of_samples, sampling_frequency*sample_length, nr_of_wave_tracks)
            indices_samples:np.ndarray      time indices of the samples at the end of the waveform samples with shape:
                                            (nr_of_samples, 1)
    """
    # 1. Calculate the indices of the samples based on the sampling frequency, the sample length, and the record length.
    wave_sample_length = int(sampling_frequency * sample_length)

    indices_samples = np.arange(wave_sample_length, record.shape[0], wave_sample_length)
    
    nr_of_samples = len(indices_samples)

    # Put samples in rows and timepoints in columns.
    indices_wave_samples = np.transpose(
        np.linspace(indices_samples - wave_sample_length, indices_samples - 1, num=wave_sample_length))
    indices_wave_samples = indices_wave_samples.reshape(nr_of_samples, wave_sample_length).astype(int)
    # End of sample indices, reshape indices_samples to be a column vector for consistency.
    indices_samples = indices_samples.reshape(nr_of_samples, 1)

    # 2. Find the wave tracks and their indices in the track_names list.
    indices_wave, track_names_wave = zip(
        *[(i, item) for i, item in enumerate(track_names) if item in ['SNUADC/ECG_II', 'SNUADC/ART', 'SNUADC/CVP']])
    tracks_wave = record[:, indices_wave]

    # Sample from all wave tracks
    samples_wave_tracks = tracks_wave[indices_wave_samples]

    return samples_wave_tracks, indices_samples
