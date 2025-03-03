import os
from typing import List, Optional, Tuple, Iterator

import numpy as np
import polars as pl
import pyedflib  # Library for reading EDF files
from scipy.signal import butter, filtfilt

class Data(Iterator[Tuple[pl.DataFrame, pl.Series]]):
    def __init__(self, root_dir: str, channels_to_include: Optional[List[str]] = None, window_size_sec: float = 2,
                 step_size_sec: float = 1, chunk_size: int = None, subjects: list[str] = None):
        self.root_dir = root_dir
        self.channels_to_include = channels_to_include
        self.chunk_size = chunk_size
        self.included_subjects = subjects
        self.current_data_chunk = dict()
        self.current_metadata_chunk = dict()
        self.current_label_chunk = []
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.generator = self.load_data()
        self.seizure_index = 0

    def __iter__(self) -> Iterator[Tuple[pl.DataFrame, pl.Series]]:
        return self.generator

    def __next__(self) -> Tuple[pl.DataFrame, pl.Series]:
        return next(self.generator)

    def load_data(self):
        """
        Generator that iterates over the directory structure and yields chunks of data.
        """
        chuck_size_left = self.chunk_size
        self.current_data_chunk = {"subject": [], "window": [], "channels": dict()}
        nb_windows_per_channel = None
        for location in os.listdir(self.root_dir):
            location_path = os.path.join(self.root_dir, location)
            if os.path.isdir(location_path):  # Ensure it's a folder
                for subject in os.listdir(location_path):
                    if self.included_subjects is not None and subject not in self.included_subjects:
                        continue
                    subject_path = os.path.join(location_path, subject)
                    if os.path.isdir(subject_path):  # Ensure it's a folder
                        f: str
                        edf_files = [f for f in os.listdir(subject_path) if f.endswith(".edf")]
                        for edf_file in edf_files:
                            base_name = os.path.splitext(edf_file)[0]
                            edf_path = os.path.join(subject_path, edf_file)
                            tsv_path = os.path.join(subject_path,
                                                    f"{base_name}_a1.tsv")

                            # Check if corresponding .tsv file exists
                            if os.path.exists(tsv_path):
                                # Load EDF file
                                edf_reader = pyedflib.EdfReader(edf_path)
                                channel_headers = edf_reader.getSignalHeaders()

                                channel_headers = self.check_channels(channel_headers)
                                n_channels = len(channel_headers)

                                # Check if chunk size is reached
                                if (chuck_size_left is not None and nb_windows_per_channel is not None and
                                        chuck_size_left <= nb_windows_per_channel):
                                    edf_reader.close()
                                    df = pl.DataFrame({
                                        "subject": self.current_data_chunk["subject"],
                                        "window": self.current_data_chunk["window"],
                                        **self.current_data_chunk["channels"]
                                    })
                                    y = pl.Series(self.current_label_chunk)
                                    self.current_data_chunk = self.current_data_chunk = {"subject": [], "window": [],
                                                                                         "channels": dict()}
                                    self.current_metadata_chunk = dict()
                                    self.current_label_chunk = []
                                    chuck_size_left = self.chunk_size
                                    yield df, y
                                    edf_reader = pyedflib.EdfReader(edf_path)

                                # Load TSV file
                                tsv_file = pl.read_csv(tsv_path, separator='\t', skip_rows=4)

                                windows, channels, labels = self.read_split_data(edf_reader, n_channels, channel_headers, tsv_file)

                                if chuck_size_left is not None:
                                    chuck_size_left -= len(windows)
                                if nb_windows_per_channel is None:
                                    nb_windows_per_channel = len(windows)

                                self.current_data_chunk["subject"].extend([subject] * len(windows))
                                self.current_data_chunk["window"].extend(windows)
                                for channel_name, channel_data in channels.items():
                                    if channel_name in self.current_data_chunk["channels"]:
                                        self.current_data_chunk["channels"][channel_name].extend(channel_data)
                                    else:
                                        self.current_data_chunk["channels"][channel_name] = channel_data
                                self.current_label_chunk.extend(labels)

                                edf_reader.close()
        yield pl.DataFrame({
            "subject": self.current_data_chunk["subject"],
            "window": self.current_data_chunk["window"],
            **self.current_data_chunk["channels"]
        }), pl.Series(self.current_label_chunk)

    def check_channels(self, channel_headers):
        """
        Check if channels to include are provided and update the channel headers accordingly to only include those channels.
        :param channel_headers: The channel headers containing the label of each channel
        :return: The updated channel headers
        """
        # Check if channels to include are provided
        if self.channels_to_include is not None:
            # Get indices of channels to include
            channels_to_include_indices = [
                i for i, header in enumerate(channel_headers) if
                header["label"] in self.channels_to_include
            ]
            # Update channel headers
            channel_headers = [channel_headers[i] for i in channels_to_include_indices]
        return channel_headers

    def read_split_data(self, edf_reader, n_channels, channel_headers, tsv_file) -> Tuple[List[int], dict, List[int]]:
        """
        Read and split the data into windows of equal size.
        :param edf_reader: The EDF reader object
        :param n_channels: The number of channels
        :param channel_headers: The channel headers, containing the information to add to the metadata
        :param tsv_file: All data present in the .tsv file in the form of a DataFrame
        :return: The windows, channels, and labels
        """
        data = dict()

        # Read data from each channel
        for i in range(n_channels):
            label = channel_headers[i]["label"]
            data[label] = edf_reader.readSignal(i)
            data[label] = self.bandpass_filter(data[label], channel_headers[i]["sample_frequency"])
            self.current_metadata_chunk[label] = channel_headers[i].copy()
            self.current_metadata_chunk[label].pop("label")

        return self.split_in_windows(data, tsv_file)

    def split_in_windows(self, data: dict, tsv_file: pl.DataFrame) -> Tuple[List[int], dict, List[int]]:
        """
        Split the data into windows of equal size.

        Parameters
        ----------
        data : dict
            The data to split.
        tsv_file : pl.DataFrame
            All data present in the .tsv file in the form of a DataFrame

        Returns
        -------
        Tuple[List[int], dict, List[int]]
            The windows, channels, and labels.
        """
        # Convert window size and step size to samples
        window_size = round(self.window_size_sec * self.current_metadata_chunk[next(iter(self.current_metadata_chunk.keys()))]["sample_frequency"])
        step_size = round(self.step_size_sec * self.current_metadata_chunk[next(iter(self.current_metadata_chunk.keys()))]["sample_frequency"])

        windows = []
        channels = {}
        labels = {}
        base_sample_rate = None
        first_channel = True
        for channel_name, channel_data in data.items():
            # Start checking seizures from the beginning for each channel
            self.seizure_index = 0

            if channel_name not in channels:
                channels[channel_name] = []

            # Check sample rate
            if base_sample_rate is None:
                base_sample_rate = self.current_metadata_chunk[channel_name]["sample_frequency"]

            # Resize the window and step size based on the sample rate
            if base_sample_rate != self.current_metadata_chunk[channel_name]["sample_frequency"]:
                window_size_i = int(window_size * self.current_metadata_chunk[channel_name]["sample_frequency"] / base_sample_rate)
                step_size_i = int(step_size * self.current_metadata_chunk[channel_name]["sample_frequency"] / base_sample_rate)
            else:
                window_size_i = window_size
                step_size_i = step_size

            # Split in windows
            for window_i, window_start in enumerate(range(0, len(channel_data) - window_size_i + 1, step_size_i)):
                label = self.get_label_window(tsv_file, window_i)

                if label is None:
                    continue

                window_data = channel_data[window_start: window_start + window_size_i]


                channels[channel_name].append(window_data)
                if first_channel:
                    windows.append(window_i)
                    labels[window_i] = label
                else:
                    assert labels[window_i] == label, "Labels are not consistent across channels."
            first_channel = False

        # Convert labels to list
        labels = list(labels.values())
        # Check windows
        # windows, channels, labels = self.check_windows(windows, channels, labels) # TODO: this removes all windows
        return windows, channels, labels


    def get_label_window(self, tsv_file: pl.DataFrame, window_i: int):
        """
        Get the label for the window based on the seizure information.
        :param tsv_file: The .tsv file containing the seizure information
        :param window_i: The index of the current window
        :return: The label for the window. 1 if the window has more than 75% overlap with a seizure, 0 if the window
        has less than 25% overlap with a seizure, and None otherwise.
        """
        window_start: float = window_i * (self.window_size_sec - self.step_size_sec)
        window_end: float = window_start + self.window_size_sec
        next_stop_seizure = tsv_file.item(row=self.seizure_index, column="stop") if tsv_file.shape[
                                                                                             0] > self.seizure_index and \
                                                                                         tsv_file.item(
                                                                                             row=self.seizure_index,
                                                                                             column="class") == "seizure" else None

        # We reached the end of the seizure. Update the next start and stop seizure to check the next seizure
        if next_stop_seizure is not None and window_start > next_stop_seizure:
            self.seizure_index += 1
            next_stop_seizure = tsv_file.item(row=self.seizure_index, column="stop") if tsv_file.shape[
                                                                                                 0] > self.seizure_index and \
                                                                                             tsv_file.item(
                                                                                                 row=self.seizure_index,
                                                                                                 column="class") == "seizure" else None

        next_start_seizure = tsv_file.item(row=self.seizure_index, column="start") if tsv_file.shape[
                                                                                          0] > self.seizure_index and \
                                                                                      tsv_file.item(
                                                                                          row=self.seizure_index,
                                                                                          column="class") == "seizure" else None
        if next_start_seizure is None:
            return 0

        overlap_window_seizure = max(0.0, min(window_end, next_stop_seizure) - max(window_start, next_start_seizure)) / self.window_size_sec
        if overlap_window_seizure > 0.75:
            return 1
        elif overlap_window_seizure < 0.25:
            return 0
        else:
            return None

    def bandpass_filter(self, data, fs, lowcut=1, highcut=25, order=4):
        """
        Apply a bandpass filter to EEG data.

        Parameters
        ----------
        data : np.ndarray
            EEG signal.
        fs : float
            Sampling frequency of the EEG signal in Hz.
        lowcut : float
            Lower cutoff frequency in Hz.
        highcut : float
            Upper cutoff frequency in Hz.
        order : int
            Order of the Butterworth filter.

        Returns
        -------
        np.ndarray
            Filtered EEG signal.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def check_windows(self, windows: list, data: dict, labels: list, lower_threshold: int = 13,
                      upper_threshold: int = 150) -> Tuple[list, dict, list]:
        """
        Check if the root-mean-square of the data is within the specified thresholds. Windows with a root-mean-square
        amplitude greater than 150 µV or less than 13 µV are removed in the default setting, because those windows
        contained high-amplitude artifacts or contained only background EEG.
        :param windows: The windows to check.
        :param data: The data to check. The keys are the channel names and the values are the window data.
        :param labels: The labels corresponding to the windows.
        :param lower_threshold: The lower threshold for the root-mean-square amplitude, in µV. Default is 13 µV.
        :param upper_threshold: The upper threshold for the root-mean-square amplitude, in µV. Default is 150 µV.
        :return: The windows, data, and labels after removing windows with a root-mean-square amplitude outside the
        specified thresholds.
        """
        # Loop over the windows in reverse order to avoid index issues
        for window_index in range(len(windows) - 1, -1, -1):
            for channel in data.keys():
                window_data = data[channel][window_index]
                rms = np.sqrt(np.mean(window_data ** 2))
                if not lower_threshold < rms < upper_threshold:
                    windows.pop(window_index)
                    labels.pop(window_index)
                    for other_channel in data.keys():
                        data[other_channel].pop(window_index)
                    break

        return windows, data, labels

