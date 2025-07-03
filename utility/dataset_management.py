import os

import numpy as np
import tensorflow as tf

from data.data import Data
from utility.paths import get_path_tfrecord


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
