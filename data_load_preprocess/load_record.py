import vitaldb
import numpy as np
import pandas as pd
from functools import reduce


def load_record(case_id, track_names, sampling_frequency=500):
    """
    Returns the record of the queried tracks for the given the case_id if all tracks contain data.
    :param case_id:int              queried case id
    :param track_names:list         queried tracks
    :param sampling_frequency:int   sampling frequency of the data in Hz
    :return:
            record:np.ndarray       record of current case ID with queried tracks, shape (nr_of_timepoints, nr_of_tracks)
    """
    # Loop through the track names and load each track
    tracks = []
    for track_name in track_names:
        track = vitaldb.vital_recs(ipath=case_id, track_names=track_name, interval=1/sampling_frequency,
                                    return_timestamp=True, return_pandas=True)
        tracks.append(track)

    # Merge all DataFrames on the 'Time' column and sort by timestamp
    record = reduce(lambda left, right: pd.merge(left, right, on='Time', how='outer'), tracks)
    record = record.sort_values(by='Time').reset_index(drop=True).drop(columns=['Time']).to_numpy()

    if len(record) == 0:  # finds cases where none of the tracks were found
        # print("---No tracks found.")
        return None
    else:  # if the tracks were found
        # if all required tracks are not empty
        if np.all(~np.isnan(record[:, :len(track_names)]).all(axis=0)):
            # print("---All tracks found!")
            return record
        else:  # if any are missing
            # print("---Missing required tracks.")
            return None
