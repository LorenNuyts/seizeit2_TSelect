import os
import sys
import time
from collections import OrderedDict

import numpy as np
import psutil
from tensorflow import keras
from data.data import Data, switch_channels

from utility.constants import Nodes
from utility.paths import get_path_preprocessed_data


# class SequentialGenerator(keras.utils.Sequence):
#     ''' Class where a keras sequential data generator is built (the data segments are continuous and aligned in time).
#
#     Args:
#         config (cls): config object with the experiment parameters
#         recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
#         segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
#                   the start and stop of the segment in seconds and the label of the segment)
#         batch_size: batch size of the generator
#         shuffle: boolean, if True, the segments are randomly mixed in every batch
#
#     '''
#
#     def __init__(self, config, recs, segments, batch_size=32, shuffle=False, verbose=True):
#
#         'Initialization'
#         self.config = config
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         self.labels = np.empty(shape=[len(segments), 2], dtype=np.float32)
#         self.verbose = verbose
#         # self.channel_selector = None
#
#         pbar = tqdm(total=len(segments) + 1, disable=not self.verbose)
#
#         count = 0
#         prev_rec = int(segments[0][0])
#
#         rec_data = Data.loadData(config.data_path, recs[prev_rec], included_channels=config.included_channels)
#         rec_data.apply_preprocess(config)
#         self.channels = rec_data.channels
#         self.data_segs = np.empty(shape=[len(segments), int(config.frame * config.fs), len(self.channels)],
#                                   dtype=np.float32)
#         # rec_data = apply_preprocess_eeg(config, rec_data)
#
#         for s in segments:
#             curr_rec = int(s[0])
#
#             if curr_rec != prev_rec:
#                 rec_data = Data.loadData(config.data_path, recs[curr_rec], included_channels=config.included_channels)
#                 rec_data.apply_preprocess(config)
#
#                 if set(rec_data.channels) != set(self.channels):
#                     rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
#                 if rec_data.channels != self.channels:
#                     rec_data.reorder_channels(self.channels)
#
#                 if rec_data.channels != self.channels and len(self.channels) != 0:
#                     print("Rec channels: ", rec_data.channels)
#                     print("self.channels: ", self.channels)
#                 assert rec_data.channels == self.channels
#                 prev_rec = curr_rec
#
#             start_seg = int(s[1] * config.fs)
#             stop_seg = int(s[2] * config.fs)
#
#             if stop_seg > len(rec_data[0]):
#                 self.data_segs[count, :, :] = np.zeros(config.fs * config.frame)
#                 # self.data_segs[count, :, 0] = np.zeros(config.fs*config.frame)
#                 # self.data_segs[count, :, 1] = np.zeros(config.fs*config.frame)
#             else:
#                 for ch_i, ch in enumerate(rec_data.channels):
#                     index_channels = self.channels.index(ch)
#                     self.data_segs[count, :, index_channels] = rec_data[ch_i][start_seg:stop_seg]
#
#             if s[3] == 1:
#                 self.labels[count, :] = [0, 1]
#             elif s[3] == 0:
#                 self.labels[count, :] = [1, 0]
#
#             count += 1
#             pbar.update(1)
#
#         self.key_array = np.arange(len(self.labels))
#
#         self.on_epoch_end()
#         config.CH = len(self.channels)
#
#     def __len__(self):
#         return len(self.key_array) // self.batch_size
#
#     def __getitem__(self, index):
#         keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
#         x, y = self.__data_generation__(keys)
#         return x, y
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             self.key_array = np.random.permutation(self.key_array)
#
#     def __data_generation__(self, keys):
#         if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
#             out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0, 2, 1, 3), self.labels[
#                 self.key_array[keys]]
#         else:
#             out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
#         return out
#
#     def change_included_channels(self, included_channels: list):
#         included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
#         assert set(included_channels).issubset(set(self.channels))
#         self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
#         self.channels = [ch for ch in self.channels if ch in included_channels]
#
#
# class SegmentedGenerator(keras.utils.Sequence):
#     ''' Class where the keras segmented data generator is built, implemented as a more efficient way to load segments that were subsampled from multiple recordings.
#
#     Args:
#         config (cls): config object with the experiment parameters
#         recs (list[list[str]]): list of recordings in the format [SUBJ-x-xxx, rxx]
#         segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
#                   the start and stop of the segment in seconds and the label of the segment)
#         batch_size: batch size of the generator
#         shuffle: boolean, if True, the segments are randomly mixed in every batch
#
#     '''
#
#     def __init__(self, config, recs, segments, batch_size=32, shuffle=True, verbose=True):
#
#         'Initialization'
#         self.config = config
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.verbose = verbose
#
#         self.data_segs = None
#         self.labels = np.empty(shape=[len(segments), 2], dtype=np.float32)
#         self.channels = []
#         segs_to_load = segments
#
#         pbar = tqdm(total=len(segs_to_load) + 1, disable=self.verbose)
#         count = 0
#
#         while segs_to_load:
#
#             curr_rec = int(segs_to_load[0][0])
#             comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]
#
#             rec_data = Data.loadData(config.data_path, recs[curr_rec], included_channels=config.included_channels)
#             rec_data.apply_preprocess(config)
#             if len(self.channels) == 0:
#                 self.channels = rec_data.channels
#             if set(rec_data.channels) != set(self.channels):
#                 rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
#             if rec_data.channels != self.channels:
#                 rec_data.reorder_channels(self.channels)
#
#             if rec_data.channels != self.channels and len(self.channels) != 0:
#                 print("Rec channels: ", rec_data.channels)
#                 print("self.channels: ", self.channels)
#             assert rec_data.channels == self.channels or len(self.channels) == 0
#
#             if self.data_segs is None:
#                 self.data_segs = np.empty(shape=[len(segments), int(config.frame * config.fs), len(self.channels)],
#                                           dtype=np.float32)
#
#             for r in comm_recs:
#                 start_seg = int(segs_to_load[r][1] * config.fs)
#                 stop_seg = int(segs_to_load[r][2] * config.fs)
#
#                 for ch_i, ch in enumerate(rec_data.channels):
#                     index_channels = self.channels.index(ch)
#                     self.data_segs[count, :, index_channels] = rec_data[ch_i][start_seg:stop_seg]
#                 # self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
#                 # self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]
#
#                 if segs_to_load[r][3] == 1:
#                     self.labels[count, :] = [0, 1]
#                 elif segs_to_load[r][3] == 0:
#                     self.labels[count, :] = [1, 0]
#
#                 count += 1
#                 pbar.update(1)
#
#             segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]
#
#         self.key_array = np.arange(len(self.labels))
#
#         self.on_epoch_end()
#         config.CH = len(self.channels)
#
#     def __len__(self):
#         return len(self.key_array) // self.batch_size
#
#     def __getitem__(self, index):
#         keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
#         x, y = self.__data_generation__(keys)
#         return x, y
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             self.key_array = np.random.permutation(self.key_array)
#
#     def __data_generation__(self, keys):
#         if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
#             out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0, 2, 1, 3), self.labels[
#                 self.key_array[keys]]
#         else:
#             out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
#         return out
#
#     def change_included_channels(self, included_channels: list):
#         included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
#         assert set(included_channels).issubset(set(self.channels))
#         self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
#         self.channels = [ch for ch in self.channels if ch in included_channels]

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
        # self.loaded_rec_data = OrderedDict()  # Use OrderedDict for caching
        # self.cache_limit = self.get_cache_limit()
        # self.current_cache_size = 0
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation__(keys)

    # def get_cache_limit(self):
    #     """Calculate cache limit as 1/3 of available memory."""
    #     free_memory = psutil.virtual_memory().available  # Get available memory in bytes
    #     return free_memory // 3  # Set limit to 1/3 of free memory

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        batch_segments = [self.segments[k] for k in keys]
        batch_data = []
        loading_time = 0
        preprocessing_time = 0
        channel_time = 0
        segmenting_time = 0
        for s in batch_segments:
            time_start = time.process_time()
            path_preprocessed_data = get_path_preprocessed_data(self.config.data_path, self.recs[int(s[0])])

            # If the preprocessed data does not exist, load and preprocess the entire recording
            if not os.path.exists(path_preprocessed_data):
                rec_data_segment = Data.loadData(self.config.data_path, self.recs[int(s[0])],
                                                 included_channels=self.config.included_channels)
                loading_time += time.process_time() - time_start
                time_start = time.process_time()
                rec_data_segment.apply_preprocess(self.config, store_preprocessed=True, recording=self.recs[int(s[0])])
                preprocessing_time += time.process_time() - time_start
                time_start = time.process_time()
                start_seg = int(s[1] * self.config.fs)
                stop_seg = int(s[2] * self.config.fs)
                # segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
                # rec_data_segment.data = rec_data_segment.data[:, start_seg:stop_seg]
                for ch_i, ch in enumerate(rec_data_segment.channels):
                    index_channels = rec_data_segment.channels.index(ch)
                    rec_data_segment.data[index_channels] = rec_data_segment[ch_i][start_seg:stop_seg]
                rec_data_segment.__segment = (s[1], s[2])  # Store segment start and stop times
                segmenting_time += time.process_time() - time_start
                time_start = time.process_time()
            # If the preprocessed data exists, load only the segment
            else:
                rec_data_segment = Data.loadSegment(self.config.data_path, self.recs[int(s[0])],
                                                    start_time=s[1], stop_time=s[2], fs=self.config.fs,
                                                    included_channels=self.config.included_channels)
                loading_time += time.process_time() - time_start
                time_start = time.process_time()

            if self.channels is None:
                self.channels = rec_data_segment.channels
            if set(rec_data_segment.channels) != set(self.channels):
                rec_data_segment.channels = switch_channels(self.channels, rec_data_segment.channels,
                                                            Nodes.switchable_nodes)
            if rec_data_segment.channels != self.channels:
                rec_data_segment.reorder_channels(self.channels)

            if rec_data_segment.channels != self.channels and len(self.channels) != 0:
                print("Rec channels:", rec_data_segment.channels)
                print("self.channels:", self.channels)
            assert rec_data_segment.channels == self.channels
            channel_time += time.process_time() - time_start
            time_start = time.process_time()

            channel_indices = [self.channels.index(ch) for ch in rec_data_segment.channels]
            segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
            segment_data[:, channel_indices] = np.array(rec_data_segment.data).T

            batch_data.append(segment_data)
            segmenting_time += time.process_time() - time_start
        batch_data = np.array(batch_data)
        if self.config.model in ['DeepConvNet', 'EEGnet']:
            batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        print(f"Loading time: {loading_time:.2f}s, Preprocessing time: {preprocessing_time:.2f}s, "
              f"Channel reordering time: {channel_time:.2f}s, Segmenting time: {segmenting_time:.2f}s")
        return batch_data, self.labels[keys]

        #     rec_key = "_".join(self.recs[int(s[0])])
        #     if rec_key in self.loaded_rec_data:
        #         rec_data = self.loaded_rec_data[rec_key]
        #     else:
        #         rec_data = Data.loadData(self.config.data_path, self.recs[int(s[0])],
        #                                  included_channels=self.config.included_channels)
        #         loading_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #         rec_data.apply_preprocess(self.config, store_preprocessed=True, recording=self.recs[int(s[0])])
        #         preprocessing_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #
        #         if self.channels is None:
        #             self.channels = rec_data.channels
        #         if set(rec_data.channels) != set(self.channels):
        #             rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
        #         if rec_data.channels != self.channels:
        #             rec_data.reorder_channels(self.channels)
        #
        #         if rec_data.channels != self.channels and len(self.channels) != 0:
        #             print("Rec channels:", rec_data.channels)
        #             print("self.channels:", self.channels)
        #         assert rec_data.channels == self.channels
        #         channel_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #
        #         # Calculate memory usage of the new entry
        #         rec_data_size = sys.getsizeof(rec_data)
        #         self.current_cache_size += rec_data_size
        #
        #         # Check cache size and remove oldest entry if necessary
        #         if self.current_cache_size > self.cache_limit:
        #             self.loaded_rec_data.popitem(last=False)  # Remove the oldest entry
        #             self.current_cache_size -= rec_data_size
        #         self.loaded_rec_data[rec_key] = rec_data  # Add new entry
        #
        #     start_seg = int(s[1] * self.config.fs)
        #     stop_seg = int(s[2] * self.config.fs)
        #     segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
        #     for ch_i, ch in enumerate(rec_data.channels):
        #         index_channels = self.channels.index(ch)
        #         segment_data[:, index_channels] = rec_data[ch_i][start_seg:stop_seg]
        #     batch_data.append(segment_data)
        #     segmenting_time += time.process_time() - time_start
        # batch_data = np.array(batch_data)
        # if self.config.model in ['DeepConvNet', 'EEGnet']:
        #     batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        # print(f"Loading time: {loading_time:.2f}s, Preprocessing time: {preprocessing_time:.2f}s, "
        #       f"Channel reordering time: {channel_time:.2f}s, Segmenting time: {segmenting_time:.2f}s")
        # self.loaded_rec_data = OrderedDict()  # Clear cache after each batch to save memory
        # return batch_data, self.labels[keys]

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
        # self.loaded_rec_data = OrderedDict()  # Use OrderedDict for caching
        # self.cache_limit = self.get_cache_limit()  # Maximum number of cached recordings
        # self.current_cache_size = 0
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation__(keys)

    # def get_cache_limit(self):
    #     """Calculate cache limit as 1/3 of available memory."""
    #     free_memory = psutil.virtual_memory().available  # Get available memory in bytes
    #     return free_memory // 3  # Set limit to 1/3 of free memory

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        batch_segments = [self.segments[k] for k in keys]
        batch_data = []
        loading_time = 0
        preprocessing_time = 0
        channel_time = 0
        segmenting_time = 0
        for s in batch_segments:
            time_start = time.process_time()
            path_preprocessed_data = get_path_preprocessed_data(self.config.data_path, self.recs[int(s[0])])

            # If the preprocessed data does not exist, load and preprocess the entire recording
            if not os.path.exists(path_preprocessed_data):
                rec_data_segment = Data.loadData(self.config.data_path, self.recs[int(s[0])],
                                                 included_channels=self.config.included_channels)
                loading_time += time.process_time() - time_start
                time_start = time.process_time()
                rec_data_segment.apply_preprocess(self.config, store_preprocessed=True, recording=self.recs[int(s[0])])
                preprocessing_time += time.process_time() - time_start
                time_start = time.process_time()
                start_seg = int(s[1] * self.config.fs)
                stop_seg = int(s[2] * self.config.fs)
                # segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
                # rec_data_segment.data = rec_data_segment.data[:, start_seg:stop_seg]
                for ch_i, ch in enumerate(rec_data_segment.channels):
                    index_channels = rec_data_segment.channels.index(ch)
                    rec_data_segment.data[index_channels] = rec_data_segment[ch_i][start_seg:stop_seg]
                rec_data_segment.__segment = (s[1], s[2])  # Store segment start and stop times
                segmenting_time += time.process_time() - time_start
                time_start = time.process_time()
            # If the preprocessed data exists, load only the segment
            else:
                rec_data_segment = Data.loadSegment(self.config.data_path, self.recs[int(s[0])],
                                                    start_time=s[1], stop_time=s[2], fs=self.config.fs,
                                                    included_channels=self.config.included_channels)
                loading_time += time.process_time() - time_start
                time_start = time.process_time()

            if self.channels is None:
                self.channels = rec_data_segment.channels
            if set(rec_data_segment.channels) != set(self.channels):
                rec_data_segment.channels = switch_channels(self.channels, rec_data_segment.channels,
                                                            Nodes.switchable_nodes)
            if rec_data_segment.channels != self.channels:
                rec_data_segment.reorder_channels(self.channels)

            if rec_data_segment.channels != self.channels and len(self.channels) != 0:
                print("Rec channels:", rec_data_segment.channels)
                print("self.channels:", self.channels)
            assert rec_data_segment.channels == self.channels
            channel_time += time.process_time() - time_start
            time_start = time.process_time()

            channel_indices = [self.channels.index(ch) for ch in rec_data_segment.channels]
            segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
            segment_data[:, channel_indices] = np.array(rec_data_segment.data).T

            batch_data.append(segment_data)
            segmenting_time += time.process_time() - time_start
        batch_data = np.array(batch_data)
        if self.config.model in ['DeepConvNet', 'EEGnet']:
            batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        print(f"Loading time: {loading_time:.2f}s, Preprocessing time: {preprocessing_time:.2f}s, "
              f"Channel reordering time: {channel_time:.2f}s, Segmenting time: {segmenting_time:.2f}s")
        return batch_data, self.labels[keys]
        #     time_start = time.process_time()
        #     rec_key = "_".join(self.recs[int(s[0])])
        #     if False:
        #         pass
        #     # if rec_key in self.loaded_rec_data:
        #     #     rec_data = self.loaded_rec_data[rec_key]
        #     else:
        #         rec_data = Data.loadData(self.config.data_path, self.recs[int(s[0])],
        #                                  included_channels=self.config.included_channels)
        #         loading_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #         rec_data.apply_preprocess(self.config, store_preprocessed=True, recording=self.recs[int(s[0])])
        #         preprocessing_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #
        #         if self.channels is None:
        #             self.channels = rec_data.channels
        #         if set(rec_data.channels) != set(self.channels):
        #             rec_data.channels = switch_channels(self.channels, rec_data.channels, Nodes.switchable_nodes)
        #         if rec_data.channels != self.channels:
        #             rec_data.reorder_channels(self.channels)
        #
        #         if rec_data.channels != self.channels and len(self.channels) != 0:
        #             print("Rec channels:", rec_data.channels)
        #             print("self.channels:", self.channels)
        #         assert rec_data.channels == self.channels
        #         channel_time += time.process_time() - time_start
        #         time_start = time.process_time()
        #
        #         # # Calculate memory usage of the new entry
        #         # rec_data_size = sys.getsizeof(rec_data)
        #         # self.current_cache_size += rec_data_size
        #         #
        #         # # Check cache size and remove oldest entry if necessary
        #         # if self.current_cache_size > self.cache_limit:
        #         #     self.loaded_rec_data.popitem(last=False)  # Remove the oldest entry
        #         #     self.current_cache_size -= rec_data_size
        #         # self.loaded_rec_data[rec_key] = rec_data  # Add new entry
        #
        #     start_seg = int(s[1] * self.config.fs)
        #     stop_seg = int(s[2] * self.config.fs)
        #     segment_data = np.zeros((self.config.frame * self.config.fs, len(self.channels)), dtype=np.float32)
        #     for ch_i, ch in enumerate(rec_data.channels):
        #         index_channels = self.channels.index(ch)
        #         segment_data[:, index_channels] = rec_data[ch_i][start_seg:stop_seg]
        #     batch_data.append(segment_data)
        #     segmenting_time += time.process_time() - time_start
        # batch_data = np.array(batch_data)
        # if self.config.model in ['DeepConvNet', 'EEGnet']:
        #     batch_data = batch_data[:, :, :, np.newaxis].transpose(0, 2, 1, 3)
        #
        # print(f"Loading time: {loading_time:.2f}s, Preprocessing time: {preprocessing_time:.2f}s, "
        #       f"Channel reordering time: {channel_time:.2f}s, Segmenting time: {segmenting_time:.2f}s")
        # # self.loaded_rec_data = OrderedDict()  # Clear cache after each batch to save memory
        # return batch_data, self.labels[keys]

    def change_included_channels(self, included_channels: list):
        included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
        assert set(included_channels).issubset(set(self.channels))
        # self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
        self.channels = [ch for ch in self.channels if ch in included_channels]


    
    
