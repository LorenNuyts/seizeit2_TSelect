import os
from typing import List

import pyedflib
import warnings

from net.utils import pre_process_ch
from utility.constants import Nodes
from utility.paths import get_path_recording, get_path_preprocessed_data


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
            included_channels (List[str]): list of modalities to include in the data object. Options are 'eeg', 'ecg', 'emg' and 'mov'.
            
        Returns:
            Data: returns a Data instance containing the data of the EDF file.
        """

        data = list()
        channels = list()
        samplingFrequencies = list()
        h5_file = get_path_preprocessed_data(data_path, recording)
        if os.path.exists(h5_file):
            return Data.load_h5(h5_file)

        edfFile = get_path_recording(data_path, recording)
        if os.path.exists(edfFile):
            with pyedflib.EdfReader(edfFile) as edf :
                samplingFrequencies.extend(edf.getSampleFrequencies())
                channels_in_file = edf.getSignalLabels()
                channels_in_file = Nodes.match_nodes(channels_in_file, included_channels)
                n = edf.signals_in_file
                for i in range(n):
                    included_channels = switch_channels(channels_in_file, included_channels, Nodes.switchable_nodes)
                    if channels_in_file[i] in included_channels:
                        data.append(edf.readSignal(i))
                        channels.append(channels_in_file[i])

                edf._close()

                samplingFrequencies = [fs for fs, ch in zip(samplingFrequencies, channels) if ch in included_channels]
                # channels = [ch for ch in channels if ch in included_channels]
                assert len([ch for ch in channels if ch not in Nodes.basic_eeg_nodes + Nodes.optional_eeg_nodes +
                            Nodes.wearable_nodes + Nodes.eeg_acc + Nodes.eeg_ears +
                            Nodes.eeg_gyr + Nodes.ecg_emg_nodes + Nodes.other_nodes + Nodes.ecg_emg_sd_acc +
                            Nodes.ecg_emg_sd_gyr]) == 0, 'Unknown channel found'
                assert len(data) == len(channels), 'Data and channels do not have the same length'
        else:
            warnings.warn('Recording ' + recording[0] + ' ' + recording[1] + ' does not contain exist')
                
        return cls(
            data,
            channels,
            samplingFrequencies,
        )

    def store_h5(self, file_path: str) -> None:
        """Store the data object in an HDF5 file.

        Args:
            file_path (str): path to the HDF5 file.
        """
        import h5py
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
            data = [h5f[channel][()] for channel in h5f.keys() if channel != 'fs']
            channels = [channel for channel in h5f.keys() if channel != 'fs']
            fs = h5f['fs'][()]
            data = cls(data, channels, fs)
            data.__preprocessed = True  # Mark as preprocessed if loaded from HDF5
            return data

    def apply_preprocess(self, config, store_preprocessed=False, recording=None) -> None:
        """
        Apply preprocessing to the data object.

        Args:
            config (Config): configuration object containing preprocessing parameters.
            store_preprocessed (bool): whether to store the preprocessed data in an HDF5 file.
            recording (List[str]): list of recording names, where the first element is the subject ID, the second is
            the recording ID and the third is the segment ID. This argument is only used if store_preprocessed is True,
            to determine the file path to store the preprocessed data.
        """
        if self.__preprocessed:
            return
        for i, channel in enumerate(self.channels):
            self.data[i], self.fs[i] = pre_process_ch(self.data[i], self.fs[i], config.fs)
        self.__preprocessed = True
        if store_preprocessed:
            preprocessed_file = get_path_preprocessed_data(config.data_path, recording)
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
    Switch the channels in the desired_channels list if the switchable_channels dictionary contains the channel to
    switch. A copy of the desired_channels list is returned with the switched channels.
    """
    missing_channels = [ch for ch in desired_channels if ch not in available_channels]
    options = {ch for ch in available_channels if ch not in desired_channels}
    non_switched_channels = set()
    result = desired_channels.copy()
    for ch in missing_channels:
        if (ch in Nodes.wearable_nodes and len({c for c in Nodes.wearable_nodes if c in available_channels}) >= 2 and
            len({c for c in Nodes.wearable_nodes if c in desired_channels}) >= 3):
            continue # Skip if 2 out of 3 wearable nodes are available
        if ch in switchable_channels:
            index = desired_channels.index(ch)
            switch_found = False
            for option in options:
                if option in switchable_channels[ch]:
                    result[index] = option
                    options.remove(option)
                    switch_found = True
                    break
            if not switch_found:
                non_switched_channels.add(ch)


    if len(non_switched_channels) > 0:
        warnings.warn(f"Could not find a suitable channel for {non_switched_channels}. The available channels are {available_channels}, "
                         f"while the requested channels are {desired_channels}")
    return result
