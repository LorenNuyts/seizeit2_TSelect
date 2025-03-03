import os
from typing import List

import numpy as np
import polars as pl
import pandas as pd

def get_recs_list(root_dir: str, locations: List[str], subjects: List[str]):
    recs = []
    for location in locations:
        for subject in os.listdir(f"{root_dir}/{location}"):
            if subject in subjects:
                path = f"{root_dir}/{location}/{subject}"
                if os.path.exists(path):
                    recs.extend([[location, rec.split("_")[0], rec.split("_")[1][:-4]] for rec in os.listdir(path) if rec.endswith(".edf")])

    return recs

def polars_to_pandas(pl_df: pl.DataFrame) -> pd.DataFrame:
    # Extract column names
    subject_col = pl_df.columns[0]  # First column (subject ID)
    windows_col = pl_df.columns[1]  # Second column (time windows)
    channel_cols = pl_df.columns[2:]  # Remaining columns (features)

    # Initialize list to store expanded rows
    expanded_rows = []

    # Iterate over Polars DataFrame rows
    for i_row, row in enumerate(pl_df.iter_rows(named=True)):
        subject_id = row[subject_col]
        # window = row[windows_col]
        channels = {col: row[col] for col in channel_cols}

        # Create t rows for each subject, one for each time step
        t = len(next(iter(channels.values())))  # Assuming all arrays have the same length t
        for i in range(t):
            expanded_rows.append(
                {"subject_id": f"{subject_id}_{i_row}", "time": i, **{col: channels[col][i] for col in channel_cols}}
            )

    # Convert expanded rows into a Pandas DataFrame
    pandas_df = pd.DataFrame(expanded_rows)

    # Set MultiIndex with `subject_id` and `time`
    pandas_df.set_index(["subject_id", "time"], inplace=True)

    return pandas_df

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
