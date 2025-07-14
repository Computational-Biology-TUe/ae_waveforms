import os
from dotenv import load_dotenv


#%% Take environment variables from .env file.
filepath_env = 'vars.env'
load_dotenv(filepath_env)
path_data = os.environ['DATA_FOLDER_PATH']

demographics = os.environ['DEMOGRAPHICS_FILE_NAME']
npt_project = os.environ['NPT_PROJECT']
npt_api_token = os.environ['NPT_API_TOKEN']


#%% Data extraction parameters

parallel_preprocessing = False

devices = [None]
track_names = ["SNUADC/ECG_II"]

use_pickle = True  # If True, the .npy files will be saved with pickle, which allows for larger files.
sampling_frequency = 500  # in Hz
sampling_rate = 1 / sampling_frequency  # in seconds

pvc_threshold = 150  # heart rate variability threshold above which a sample is believed to contain PVCs.

butter_order = 4
# Butterworth filter parameters for ECG signals (in Hz)
butter_hp_ecg = 1   # 1
butter_lp_ecg = 30  # 30

sample_length = 20  # Length of the DL input samples in seconds
