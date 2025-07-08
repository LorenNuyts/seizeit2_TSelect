import os
from typing import List

import numpy as np
import pyedflib
import warnings

import tensorflow as tf

from net.utils import pre_process_ch
from utility.constants import Nodes
from utility.paths import get_path_recording, get_path_preprocessed_data, get_path_tfrecord


class Data:
    def __init__(
        self,
        data,
        channels: List[str],
        fs: List[int],
    ):
        """Initiate a Data instance

        Args:
            data (List(NDArray[Shape['1, *'], float])): a list of data arrays. Each channel's data is stored as an entry in the list as a data array that stores the samples in time.
            channels (List[str]): tuple of channels as strings.
            fs (List[int]): Sampling frequency of each channel.
        """
        self.data = data
        self.channels = channels
        self.fs = fs
        self.__preprocessed = False
        self.__segment = None

    def get_duration(self) -> float:
        """Get the duration of the data object in seconds.

        Returns:
            float: duration of the data object in seconds.
        """
        if self.__segment is not None:
            return self.__segment[1] - self.__segment[0]
        else:
            return len(self.data[0]) / self.fs[0] if self.fs else 0.0

    @classmethod
    def loadData(
        cls,
        data_path: str,
        recording: List[str],
        included_channels: List[str],
    ):
        """Instantiate a data object from an EDF file.

        Args:
            data_path (str): path to EDF file.
            recording (List[str]): list of recording names, where the first element is the subject ID, the second is the recording ID and the third is the segment ID.
            included_channels (List[str]): list of channels to include in the data object. If a channel is not present in the EDF file, it will be switched to a similar channel if possible.
            
        Returns:
            Data: returns a Data instance containing the data of the EDF file.
        """

        data = list()
        channels = list()
        samplingFrequencies = list()
        h5_file = get_path_preprocessed_data(data_path, recording)
        if os.path.exists(h5_file):
            # pass
            return Data.load_h5(h5_file)

        edfFile = get_path_recording(data_path, recording)
        if os.path.exists(edfFile):
            with pyedflib.EdfReader(edfFile) as edf :
                all_samplingFrequencies = edf.getSampleFrequencies()
                samplingFrequencies.extend(edf.getSampleFrequencies())
                channels_in_file = edf.getSignalLabels()
                standardized_channels_in_file = switch_channels(channels_in_file, included_channels, Nodes.switchable_nodes)
                standardized_included_channels = Nodes.match_nodes(included_channels, Nodes.all_nodes())
                for ch in sorted(standardized_included_channels):
                    if ch in standardized_channels_in_file:
                        ix = standardized_channels_in_file.index(ch)
                        data.append(edf.readSignal(ix))
                        channels.append(ch)
                        samplingFrequencies.append(all_samplingFrequencies[ix])
                # channels_in_file = Nodes.match_nodes(channels_in_file, Nodes.all_nodes())
                # for ch in sorted(included_channels):
                # n = edf.signals_in_file
                # for i in range(n):
                #     included_channels = switch_channels(channels_in_file, included_channels, Nodes.switchable_nodes)
                #     if channels_in_file[i] in included_channels:
                #         data.append(edf.readSignal(i))
                #         channels.append(channels_in_file[i])

                edf._close()

                # samplingFrequencies = [fs for fs, ch in zip(samplingFrequencies, channels) if ch in included_channels]
                # channels = [ch for ch in channels if ch in included_channels]
                assert len([ch for ch in channels if ch not in Nodes.basic_eeg_nodes + Nodes.optional_eeg_nodes +
                            Nodes.wearable_nodes + Nodes.eeg_acc + Nodes.eeg_ears +
                            Nodes.eeg_gyr + Nodes.ecg_emg_nodes + Nodes.other_nodes + Nodes.ecg_emg_sd_acc +
                            Nodes.ecg_emg_sd_gyr]) == 0, 'Unknown channel found'
                assert len(data) == len(channels), 'Data and channels do not have the same length'
        else:
            warnings.warn('Recording ' + recording[0] + ' ' + recording[1] + ' does not contain exist')

        data_object = cls(data, channels, samplingFrequencies)
        data_object.__preprocessed = False
        data_object.__segment = None
        return data_object

    @classmethod
    def loadSegment(cls, data_path, recording, start_time, stop_time, fs: int, included_channels=None):
        """Load a segment of the data object.

        Args:
            data_path (str): path to EDF file.
            recording (List[str]): list of recording names, where the first element is the subject ID, the second is the recording ID and the third is the segment ID.
            start_time (float): start time of the segment in seconds.
            stop_time (float): stop time of the segment in seconds.
            fs (int): sampling frequency of the segment.
            included_channels (List[str]): list of channels to include in the data object. If a channel is not present in the EDF file, it will be switched to a similar channel if possible.

        Returns:
            Data: returns a Data instance containing the segment of the data object.
        """

        h5_file = get_path_preprocessed_data(data_path, recording)
        if os.path.exists(h5_file):
            return cls.loadSegment_h5(h5_file, start_time, stop_time, fs, included_channels)
        else:
            warnings.warn("No preprocessed data found. Loading entire EDF file, preprocessing and storing it... ")
            data_object = cls.loadData(data_path, recording, included_channels)
            data_object.apply_preprocess(fs, data_path=data_path, store_preprocessed=False, recording=recording)
            return cls.loadSegment_h5(h5_file, start_time, stop_time, fs, included_channels)
            # raise ValueError("Segments can only be loaded from preprocessed data. Please save the preprocessed data first.")


    def store_h5(self, file_path: str) -> None:
        """Store the data object in an HDF5 file.

        Args:
            file_path (str): path to the HDF5 file.
        """
        if self.__segment is not None:
            raise ValueError("Cannot store a segment of data in an HDF5 file.")
        import h5py
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, 'w') as h5f:
            for i, channel in enumerate(self.channels):
                h5f.create_dataset(channel, data=self.data[i], compression='gzip')
            h5f.create_dataset('fs', data=self.fs, compression='gzip')
    
    @classmethod
    def load_h5(cls, file_path: str):
        """Load the data object from an HDF5 file.

        Args:
            file_path (str): path to the HDF5 file.
        """
        import h5py
        with h5py.File(file_path, 'r') as h5f:
            try:
                data = []
                channels = []
                for channel in h5f.keys():
                    if channel != 'fs':
                        data.append(h5f[channel][()])
                        channels.append(channel)
                # data = [h5f[channel][()] for channel in h5f.keys() if channel != 'fs']
                # channels = [channel for channel in h5f.keys() if channel != 'fs']
                fs = h5f['fs'][()]
                data = cls(data, channels, fs)
                data.__preprocessed = True  # Mark as preprocessed if loaded from HDF5
                data.__segment = None
                return data
            except OSError:
                raise ValueError(f"Could not load data from {file_path}. The file might be corrupted or not in the expected format.")

    @classmethod
    def loadSegment_h5(cls, file_path: str, start_time: float, stop_time: float, fs: int, included_channels: List[str] = None):
        """Load a segment of the data object from an HDF5 file.

        Args:
            file_path (str): path to the HDF5 file.
            start_time (float): start time of the segment in seconds.
            stop_time (float): stop time of the segment in seconds.
            fs (int): sampling frequency of the segment.
            included_channels (List[str]): list of channels to include in the data object. If a channel is not present
            in the HDF5 file, it will be switched to a similar channel if possible. If None, all channels listed in the
            Nodes class will be included.
        """
        if included_channels is None:
            included_channels = Nodes.all_nodes()
        import h5py
        with h5py.File(file_path, 'r') as h5f:
            start_sample = int(start_time * fs)
            stop_sample = int(stop_time * fs)
            data = []
            channels = []
            samplingFrequencies = []
            all_samplingFrequencies = h5f['fs'][()]
            channels_in_file = list(h5f.keys())
            standardized_channels_in_file = switch_channels(channels_in_file, included_channels, Nodes.switchable_nodes)
            standardized_included_channels = Nodes.match_nodes(included_channels, Nodes.all_nodes())
            for ch in sorted(standardized_included_channels):
                if ch in standardized_channels_in_file:
                    ix = standardized_channels_in_file.index(ch)
                    data.append(h5f[channels_in_file[ix]][start_sample:stop_sample])
                    channels.append(ch)
                    samplingFrequencies.append(all_samplingFrequencies[ix])
            # for channel in h5f.keys():
            #     if channel != 'fs':
            #         segment = h5f[channel][start_sample:stop_sample]
            #         data.append(segment)
            #         channels.append(channel)

            segment = cls(data, channels, samplingFrequencies)
            segment.__preprocessed = True  # Mark as preprocessed if loaded from HDF5
            segment.__segment = (start_time, stop_time)
            return segment

    def apply_preprocess(self, fs, store_preprocessed=False, data_path=None, recording=None) -> None:
        """
        Apply preprocessing to the data object.

        Args:
            fs (int): target sampling frequency to which the data will be resampled.
            store_preprocessed (bool): whether to store the preprocessed data in an HDF5 file.
            data_path (str): path to the data directory where the preprocessed data will be stored.
            recording (List[str]): list of recording names, where the first element is the subject ID, the second is
            the recording ID and the third is the segment ID. This argument is only used if store_preprocessed is True,
            to determine the file path to store the preprocessed data.
        """
        if self.__preprocessed:
            return
        for i, channel in enumerate(self.channels):
            self.data[i], self.fs[i] = pre_process_ch(self.data[i], self.fs[i], fs)
        self.__preprocessed = True
        if store_preprocessed and self.__segment is None:
            if data_path is None:
                raise ValueError("data_path must be provided if store_preprocessed is True.")
            preprocessed_file = get_path_preprocessed_data(data_path, recording)
            self.store_h5(preprocessed_file)

    def __getitem__(self, index):
        return self.data[index]

    def reorder_channels(self, channels: List[str]):
        """Reorder the channels in the data object to match the order of the channels in the channels list.

        Args:
            channels (List[str]): list of channels to reorder the data object to.
        """
        new_data = []
        new_fs = []
        for ch in channels:
            if ch in self.channels:
                new_data.append(self.data[self.channels.index(ch)])
                new_fs.append(self.fs[self.channels.index(ch)])
            else:
                raise ValueError(f"Channel {ch} not found in data object. Available channels are: {self.channels}")
        self.data = new_data
        self.channels = channels
        self.fs = new_fs


def switch_channels(available_channels: list[str], desired_channels: list[str], switchable_channels: dict) -> list[str]:
    """
    Switch the channels in the available_channels list if the switchable_channels dictionary contains the channel to
    switch. A copy of the desired_channels list is returned with the switched channels.
    """
    # Format both sets of channels to eliminate differences in writing style
    available_channels = Nodes.match_nodes(available_channels, Nodes.all_nodes())
    desired_channels = Nodes.match_nodes(desired_channels, Nodes.all_nodes())

    missing_channels = [ch for ch in desired_channels if ch not in available_channels]
    options = {ch for ch in available_channels if ch not in desired_channels}
    non_switched_channels = set()
    result = available_channels.copy()
    for ch in missing_channels:
        # if (ch in Nodes.wearable_nodes and len({c for c in Nodes.wearable_nodes if c in available_channels}) >= 2 and
        #     len({c for c in Nodes.wearable_nodes if c in desired_channels}) >= 3):
        #     continue # Skip if 2 out of 3 wearable nodes are available
        switch_found = False
        if ch in switchable_channels:
            for option in options:
                if option in switchable_channels[ch]:
                    index = available_channels.index(option)
                    result[index] = ch
                    options.remove(option)
                    switch_found = True
                    break
        if not switch_found:
            non_switched_channels.add(ch)


    if len(non_switched_channels) > 0:
        warnings.warn(f"Could not find a suitable channel for {non_switched_channels}. The available channels are {available_channels}, "
                         f"while the requested channels are {desired_channels}")
    return result


def serialize_example(segment_data, label):
    """Serializes one segment + label into a TFRecord-compatible Example."""
    feature = {
        "segment": tf.train.Feature(bytes_list=tf.train.BytesList(value=[segment_data.tobytes()])),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_single_tfrecord(config, recs, segment):
    rec_index, start_time, stop_time, label_val = segment
    recording = recs[int(rec_index)]
    tfrecord_path = get_path_tfrecord(config.data_path, recording, start_time, stop_time)

    if os.path.exists(tfrecord_path):
        return

    # Load the preprocessed segment
    s = Data.loadSegment(config.data_path, recording,
                               start_time=start_time,
                               stop_time=stop_time,
                               fs=config.fs,
                               included_channels=config.included_channels)
    # Build data tensor
    segment_data = np.stack(s.data, axis=1)  # shape (T, CH)
    segment_data = segment_data.astype(np.float32)
    segment_data = segment_data[:, :, np.newaxis]  # shape (T, CH, 1)

    # Check that the segment data has the expected shape
    assert segment_data.shape[1] == 21, \
        (f"Segment data shape mismatch: {segment_data.shape[1]} channels found, expected {len(config.included_channels)} "
         f"channels.")
    # Check that the channels are ordered alphabetically
    assert np.all(s.channels == sorted(config.included_channels)), \
        "Channels do not match the ordered list of included channels. Expected: {}, Found: {}".format(
            sorted(config.included_channels), s.channels)
    # Check that none of T3, T4, T5, T6 or BTEright are present
    assert not any(channel in s.channels for channel in ['T3', 'T4', 'T5', 'T6', Nodes.BTEright]), \
        "Channels T3, T4, T5, T6 or BTEright should not be present in the data. Found: {}".format(
            [channel for channel in s.channels if channel in ['T3', 'T4', 'T5', 'T6', Nodes.BTEright]])

    # Transpose if model requires it
    if config.model in ['DeepConvNet', 'EEGnet']:
        segment_data = segment_data.transpose(1, 0, 2)  # (CH, T, 1)
    # Build label vector
    label = [1.0, 0.0] if label_val == 0 else [0.0, 1.0]
    # Write example
    example = serialize_example(segment_data, label)

    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        writer.write(example)
