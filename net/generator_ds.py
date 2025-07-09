import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from data.data import Data, switch_channels, create_single_tfrecord

from utility.constants import Nodes
from utility.paths import get_path_preprocessed_data, get_path_tfrecord


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
                rec_data_segment.apply_preprocess(self.config.fs, data_path=self.config.data_path, store_preprocessed=True, recording=self.recs[int(s[0])])
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
                rec_data_segment.apply_preprocess(self.config.fs, data_path=self.config.data_path, store_preprocessed=True, recording=self.recs[int(s[0])])
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

    def change_included_channels(self, included_channels: list):
        included_channels = switch_channels(self.channels, included_channels, Nodes.switchable_nodes)
        assert set(included_channels).issubset(set(self.channels))
        # self.data_segs = self.data_segs[:, :, [i for i, ch in enumerate(self.channels) if ch in included_channels]]
        self.channels = [ch for ch in self.channels if ch in included_channels]

def segment_generator(config, recs, segments, labels, shuffle=True):
    channels = None
    key_array = np.arange(len(labels))
    while True:  # repeat forever (or use .repeat() in tf.data)
        if shuffle:
            key_array = np.random.permutation(key_array)
        for idx in key_array:
            s = segments[idx]
            y = labels[idx]

            path_preprocessed_data = get_path_preprocessed_data(config.data_path, recs[int(s[0])])

            if not os.path.exists(path_preprocessed_data):
                rec_data_segment = Data.loadData(config.data_path, recs[int(s[0])],
                                                 included_channels=config.included_channels)
                rec_data_segment.apply_preprocess(config.fs, data_path=self.config.data_path, store_preprocessed=True, recording=recs[int(s[0])])
                start_seg = int(s[1] * config.fs)
                stop_seg = int(s[2] * config.fs)
                for ch_i, ch in enumerate(rec_data_segment.channels):
                    index_channels = rec_data_segment.channels.index(ch)
                    rec_data_segment.data[index_channels] = rec_data_segment[ch_i][start_seg:stop_seg]
                rec_data_segment.__segment = (s[1], s[2])
            else:
                rec_data_segment = Data.loadSegment(config.data_path, recs[int(s[0])],
                                                    start_time=s[1], stop_time=s[2], fs=config.fs,
                                                    included_channels=config.included_channels)

            # Reorder channels
            if channels is None:
                channels = rec_data_segment.channels
            if set(rec_data_segment.channels) != set(channels):
                rec_data_segment.channels = switch_channels(channels, rec_data_segment.channels,
                                                            Nodes.switchable_nodes)
            if rec_data_segment.channels != channels:
                rec_data_segment.reorder_channels(channels)

            # Extract data
            channel_indices = [channels.index(ch) for ch in rec_data_segment.channels]
            segment_data = np.zeros((config.frame * config.fs, len(channels)), dtype=np.float32)
            segment_data[:, channel_indices] = np.array(rec_data_segment.data).T

            segment_data = segment_data[:, :, np.newaxis]
            if config.model in ['DeepConvNet', 'EEGnet']:
                segment_data = segment_data.transpose(1, 0, 2)

            yield segment_data, y

def build_segment_dataset(config, recs, segments, batch_size=32, shuffle=True):
    labels = np.array([[1, 0] if s[3] == 0 else [0, 1] for s in segments], dtype=np.float32)

    output_signature = (
        tf.TensorSpec(shape=(config.frame * config.fs, config.CH, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: segment_generator(config, recs, segments, labels, shuffle),
        output_signature=output_signature
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = math.ceil(len(segments) / batch_size)
    return dataset, steps_per_epoch

def parse_example(example_proto, config, channel_indices=None):
    features = {
        "segment": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([2], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    segment_shape = (config.frame * config.fs, len(config.included_channels), 1)
    segment_data = tf.io.decode_raw(parsed["segment"], tf.float32)
    segment_data = tf.reshape(segment_data, segment_shape)

    # Dynamically select specific channels, if requested
    if channel_indices is not None:
        selected_channels = tf.constant(channel_indices, dtype=tf.int32)
        segment_data = tf.gather(segment_data, selected_channels, axis=1)

    if config.model in ['DeepConvNet', 'EEGnet']:
        segment_data = tf.transpose(segment_data, perm=[1, 0, 2])

    return segment_data, parsed["label"]

def build_tfrecord_dataset(config, recs, segments, batch_size=32, shuffle=True, progress_bar=True, channel_indices=None):
    # Generate TFRecord paths for each segment
    tfrecord_files = []
    for s in tqdm(segments, disable=not progress_bar, desc="Preparing TFRecord files"):
        rec_idx, start, stop, _ = s
        path = get_path_tfrecord(config.data_path, recs[int(rec_idx)], start, stop)
        tfrecord_files.append(path)
        if not os.path.exists(path):
            create_single_tfrecord(config, recs, s)

    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)

    dataset = dataset.map(lambda x: parse_example(x, config, channel_indices),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = math.ceil(len(tfrecord_files) / batch_size)
    return dataset, steps_per_epoch
