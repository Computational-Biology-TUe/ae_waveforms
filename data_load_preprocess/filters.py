import numpy as np
from scipy.signal import butter, filtfilt


def zscore(data):
    """
    Z-score normalization of the input data.
    :param data:np.ndarray          data to be normalized
    :return:
           np.ndarray               normalized data
    """
    return (data - np.nanmean(data)) / np.nanstd(data)


def butterworth(data, btype, cutoff_frequency, sampling_frequency=500, filter_order=4):
    """
    Performs a Butterworth filter (lowpass or highpass) with a predefined cutoff frequency to smooth the input signal.
    The filter handles NaN segments by splitting the data into segments before and after the NaN segment and applying the
    filter to each segment separately. Afterward the segments are recombined at their original positions.
    :param data:np.ndarray          contains the data to be smoothened
    :param btype:string             type of filter to be performed, either 'low' or 'high'
    :param cutoff_frequency:int     cutoff value in Hz
    :param sampling_frequency:int   sampling frequency of the data in Hz
    :param filter_order:int         order of the butterworth filter.
    :return:
           np.ndarray               the filtered data
    """
    # Design Butterworth filter
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    [b, a] = butter(N=filter_order, Wn=normal_cutoff, btype=btype, analog=False)

    # Create output array filled with NaNs
    filtered_data = np.full_like(data, np.nan)

    # Find non-NaN segments
    isnan = np.isnan(data)
    segment_start = np.where((isnan[:-1] & ~isnan[1:]))[0] + 1    # End of NaN
    segment_end = np.where((~isnan[:-1] & isnan[1:]))[0] + 1  # Start of NaN

    # Handle case where first or last sample is not NaN
    if not isnan[0]:
        segment_start = np.insert(segment_start, 0, 0)
    if not isnan[-1]:
        segment_end = np.append(segment_end, len(data))

    # Filter each non-NaN segment separately
    for start, end in zip(segment_start, segment_end):
        # If the segment is too short, skip filtering
        if end - start <= 3 * max(len(a), len(b)):
            filtered_data[start:end] = data[start:end]
        else:
            filtered_data[start:end] = filtfilt(b, a, data[start:end])

    return filtered_data
