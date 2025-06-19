import os
from typing import List

import numpy as np
import pandas as pd

def get_recs_list(root_dir: str, locations: List[str], subjects: List[str]):
    """
    Get a list of all recordings in the specified locations and subjects. Each element in the list is a list with the
    location, subject and recording ID.
    """
    recs = []
    for location in locations:
        for subject in os.listdir(f"{root_dir}/{location}"):
            if subject in subjects:
                path = f"{root_dir}/{location}/{subject}"
                if os.path.exists(path):
                    recs.extend([[location, rec.split("_")[0], rec.split("_")[1][:-4]] for rec in os.listdir(path) if rec.endswith(".edf")])

    return recs

def smooth_predictions(predictions: np.ndarray, nb_considered_windows: int = 10, nb_required_positive_values: int = 8) \
        -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    if len(predictions) < nb_considered_windows:
        return np.zeros_like(predictions)

    # Create rolling windows of size `nb_considered_windows`
    windows = sliding_window_view(predictions, nb_considered_windows)

    # Compute the sum of 1s in each window
    valid_mask: np.ndarray = np.sum(windows, axis=1) >= nb_required_positive_values

    # Expand valid_mask back to original size (apply same label to all in window)
    smoothed = np.zeros_like(predictions)

    for i in range(len(valid_mask)):
        if valid_mask[i]:  # If the window satisfies the condition, set all x values to 1
            smoothed[i: i + nb_considered_windows] = 1

    return smoothed
    # mask = predictions.rolling(window=nb_considered_windows, min_periods=nb_considered_windows).sum() >= nb_required_positive_values
    # # Expand mask back to original size
    # return mask.repeat(nb_considered_windows).iloc[:len(predictions)].astype(int)
