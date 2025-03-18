import os
from typing import List

import pyedflib
import warnings

from net.utils import pre_process_ch
from utility.constants import Nodes


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

        edfFile = os.path.join(data_path, recording[0], recording[1], f"{recording[1]}_{recording[2]}.edf")
        if os.path.exists(edfFile):
            with pyedflib.EdfReader(edfFile) as edf :
                samplingFrequencies.extend(edf.getSampleFrequencies())
                channels.extend(edf.getSignalLabels())
                n = edf.signals_in_file
                for i in range(n):
                    included_channels = switch_channels(channels, included_channels, Nodes.switchable_nodes)
                    if channels[i] in included_channels:
                        data.append(edf.readSignal(i))
                edf._close()

                samplingFrequencies = [fs for fs, ch in zip(samplingFrequencies, channels) if ch in included_channels]
                channels = [ch for ch in channels if ch in included_channels]
                assert len([ch for ch in channels if ch not in Nodes.basic_eeg_nodes  + Nodes.optional_eeg_nodes +
                            Nodes.wearable_nodes + Nodes.eeg_acc +
                            Nodes.eeg_gyr + Nodes.ecg_emg_nodes + Nodes.other_nodes + Nodes.ecg_emg_acc +
                            Nodes.ecg_emg_gyr]) == 0, 'Unknown channel found'
                assert len(data) == len(channels), 'Data and channels do not have the same length'
        else:
            warnings.warn('Recording ' + recording[0] + ' ' + recording[1] + ' does not contain exist')
                
        return cls(
            data,
            channels,
            samplingFrequencies,
        )

    def apply_preprocess(self, config) -> None:
        for i, channel in enumerate(self.channels):
            self.data[i], self.fs[i] = pre_process_ch(self.data[i], self.fs[i], config.fs)

    def __getitem__(self, index):
        return self.data[index]


def switch_channels(channels: list[str], included_channels: list[str], switchable_channels: dict) -> list[str]:
    """
    Switch the channels in the included_channels list if the switchable_channels dictionary contains the channels to
    switch. A copy of the included_channels list is returned with the switched channels.
    """
    missing_channels = [ch for ch in included_channels if ch not in channels]
    result = included_channels.copy()
    for ch in missing_channels:
        if ch in switchable_channels:
            index = included_channels.index(ch)
            result[index] = switchable_channels[ch]
    return result
