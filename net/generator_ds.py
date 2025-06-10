import numpy as np
from tensorflow import keras
from tqdm import tqdm
from data.data import Data, switch_channels

from utility.constants import Nodes


class SequentialGenerator(keras.utils.Sequence):
    ''' Class where a keras sequential data generator is built (the data segments are continuous and aligned in time).

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=False, verbose=True):
        super().__init__()
        self.config = config
        self.recs = recs
        self.segments = segments
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.channels = None
        self.labels = np.array([[1, 0] if s[3] == 0 else [0, 1] for s in segments], dtype=np.float32)
        self.key_array = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation__(keys)

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        batch_segments = [self.segments[k] for k in keys]
        batch_data = []
        for s in batch_segments:
            rec_data = Data.loadData(self.config.data_path, self.recs[int(s[0])],
                                     included_channels=self.config.included_channels)
            rec_data.apply_preprocess(self.config)

            if set(rec_data.channels) != set(self.channels):
                rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
            if rec_data.channels != self.channels:
                rec_data.reorder_channels(self.channels)

            if rec_data.channels != self.channels and len(self.channels) != 0:
                print("Rec channels:", rec_data.channels)
                print("self.channels:", self.channels)
            assert rec_data.channels == self.channels

            # if self.channels is None:
            #     self.channels = rec_data.channels
            start_seg = int(s[1] * self.config.fs)
            stop_seg = int(s[2] * self.config.fs)
            segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
            for ch_i, ch in enumerate(rec_data.channels):
                index_channels = self.channels.index(ch)
                segment_data[:, index_channels] = rec_data[ch_i][start_seg:stop_seg]
            batch_data.append(segment_data)
        batch_data = np.array(batch_data)
        if self.config.model in ['DeepConvNet', 'EEGnet']:
            batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        return batch_data, self.labels[keys]

    def change_included_channels(self, included_channels: list):
        included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
        assert set(included_channels).issubset(set(self.channels))
        # self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
        self.channels = [ch for ch in self.channels if ch in included_channels]


class SegmentedGenerator(keras.utils.Sequence):
    ''' Class where the keras segmented data generator is built, implemented as a more efficient way to load segments that were subsampled from multiple recordings.

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [SUBJ-x-xxx, rxx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=True, verbose=True):
        super().__init__()
        self.config = config
        self.recs = recs
        self.segments = segments
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.channels = None
        self.labels = np.array([[1, 0] if s[3] == 0 else [0, 1] for s in segments], dtype=np.float32)
        self.key_array = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation__(keys)

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        batch_segments = [self.segments[k] for k in keys]
        batch_data = []
        for s in batch_segments:
            rec_data = Data.loadData(self.config.data_path, self.recs[int(s[0])],
                                     included_channels=self.config.included_channels)
            rec_data.apply_preprocess(self.config)

            if set(rec_data.channels) != set(self.channels):
                rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
            if rec_data.channels != self.channels:
                rec_data.reorder_channels(self.channels)

            if rec_data.channels != self.channels and len(self.channels) != 0:
                print("Rec channels:", rec_data.channels)
                print("self.channels:", self.channels)
            assert rec_data.channels == self.channels
            # if self.channels is None:
            #     self.channels = rec_data.channels
            start_seg = int(s[1] * self.config.fs)
            stop_seg = int(s[2] * self.config.fs)
            segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
            for ch_i, ch in enumerate(rec_data.channels):
                index_channels = self.channels.index(ch)
                segment_data[:, index_channels] = rec_data[ch_i][start_seg:stop_seg]
            batch_data.append(segment_data)
        batch_data = np.array(batch_data)
        if self.config.model in ['DeepConvNet', 'EEGnet']:
            batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        return batch_data, self.labels[keys]

    def change_included_channels(self, included_channels: list):
        included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
        assert set(included_channels).issubset(set(self.channels))
        # self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
        self.channels = [ch for ch in self.channels if ch in included_channels]


    
    
