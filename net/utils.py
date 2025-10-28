from typing import List

import tensorflow as tf
import numpy as np
from scipy import signal

from utility.constants import Nodes


def set_gpu():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)


def decay_schedule(epoch, lr):
    if lr > 1e-5:
        if (epoch + 1) % 10 == 0:
            lr = lr / 2
        
    return lr


#######################################
############### metrics ###############

def get_sens_FA_score(y_true, y_pred, fs=1/2, th=0.5):
    pass

#### Pre-process EEG data

def apply_preprocess_eeg(config, rec):

    idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']
    idx_cross = [i for i, c in enumerate(rec.channels) if c == 'CROSStop SD']
    if not idx_cross:
        idx_cross = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']

    ch_focal, _ = pre_process_ch(rec.data[idx_focal[0]], rec.fs[idx_focal[0]], config.fs)
    ch_cross, _ = pre_process_ch(rec.data[idx_cross[0]], rec.fs[idx_cross[0]], config.fs)
        
    # ch_focal = (ch_focal - np.mean(ch_focal))/np.std(ch_focal)
    # ch_cross = (ch_cross - np.mean(ch_cross))/np.std(ch_cross)

    return [ch_focal, ch_cross]


def pre_process_ch(ch_data, fs_data, fs_resamp):

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))
    
    b, a = signal.butter(4, 0.5/(fs_resamp/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs_resamp/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs_resamp/2), 50.5/(fs_resamp/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


def rereference_average_signal(data, channels: List[str],
                               exclude_channels_for_average: List[str] = Nodes.prefrontal_nodes + Nodes.wearable_nodes,
                               exclude_channels_for_rereference: List[str] = Nodes.wearable_nodes) -> List[np.ndarray]:
    """
    Rereference the EEG signal using the average of all channels, excluding specified channels.
    Args:
        data(List(NDArray[Shape['1, *'], float])): a list of data arrays. Each channel's data is stored as an
                                                   entry in the list as a data array that stores the samples in time.
        channels: List of channel names corresponding to the data arrays.
        exclude_channels_for_average: List of channel names to exclude from the average reference calculation.
                          Default is Nodes.prefrontal_nodes + Nodes.wearable_nodes.
        exclude_channels_for_rereference: List of channel names to exclude from the rereferencing process.
                            Default is Nodes.wearable_nodes.
    Returns:
        rereferenced_data: List of 1D numpy arrays, each representing the rereferenced EEG signal.
    """
    assert len(data) == len(channels), "Data and channels must have the same length."
    include_indices = [i for i, ch in enumerate(channels) if ch not in exclude_channels_for_average]
    if not include_indices:
        raise ValueError("No channels left to compute average reference after excluding specified channels.")
    included_data = np.array([data[i] for i in include_indices])
    avg_reference = np.mean(included_data, axis=0)
    to_rereference = [i for i, ch in enumerate(channels) if ch not in exclude_channels_for_rereference]
    rereferenced_data = []
    for i in range(len(data)):
        if i in to_rereference:
            rereferenced_data.append(data[i] - avg_reference)
        else:
            rereferenced_data.append(data[i])  # Keep original if excluded from reference
    return rereferenced_data
