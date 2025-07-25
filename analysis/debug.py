import argparse
import os
import pickle
import tensorflow as tf
from tqdm import tqdm

from analysis.dataset import dataset_stats
from data.cross_validation import multi_objective_grouped_stratified_cross_validation
from net.DL_config import get_base_config
from net.generator_ds import build_tfrecord_dataset, parse_example
from net.key_generator import generate_data_keys_sequential_window, generate_data_keys_sequential, \
    generate_data_keys_subsample
from utility import get_recs_list
from utility.constants import Locations, SEED
from data.data import create_single_tfrecord
from utility.paths import get_paths_segments_train, get_paths_segments_val, get_path_tfrecord

base_ = os.path.dirname(os.path.realpath(__file__))

def inspect_channels():
    config = get_base_config(os.path.join(base_, ".."), locations=[Locations.karolinska, Locations.freiburg,
                                                                   Locations.leuven_adult])
    recs = [# ['University_Hospital_Leuven_Pediatric', 'SUBJ-1b-315', 'r4'],
            # ['University_Hospital_Leuven_Adult', 'SUBJ-1a-006', 'r10'],
            ['Karolinska_Institute', 'SUBJ-6-402', 'r11'],
            ]
    # segments = generate_data_keys_sequential(config, recs, 6 * 60)
    # segments = generate_data_keys_sequential_window(config, recs, 6 * 60)
    # segments = generate_data_keys_subsample(config, recs)
    segments = [[  0, 12398., 12400.,     0.]]
    for s in tqdm(segments, desc="Writing TFRecord"):
        create_single_tfrecord(config, recs, s)

def negative_dimensions():
    config = get_base_config(os.path.join(base_, ".."), locations=[Locations.leuven_adult])
    val_recs_list = [#['Karolinska_Institute', 'SUBJ-6-430', 'r26'],
                     # ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r1'],
                     # ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r2'],
                     # ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r3'],
                     ['University_Hospital_Leuven_Adult', 'SUBJ-1a-006', 'r10'],
                     ]
    val_segments = generate_data_keys_sequential(config, val_recs_list, 6 * 60)

def ts_reshape_error():
    config = get_base_config(os.path.join(base_, ".."), locations=sorted([Locations.karolinska, Locations.freiburg,
                                                                          Locations.leuven_adult, Locations.leuven_pediatric,
                                                                          Locations.aachen, Locations.coimbra]))
    # recs = [['University_Hospital_Leuven_Adult', 'SUBJ-1a-152', 'r7']]
    # segment = [ 0, 3282, 3284 ,   0]
    # create_single_tfrecord(config, recs, segment)
    info_per_group = dataset_stats(config.data_path, os.path.join("..", config.save_dir, "dataset_stats"), config.locations)

    CV_generator = multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                           id_column='subject',
                                                                           n_splits=config.n_folds,
                                                                           train_size=config.train_percentage,
                                                                           val_size=config.validation_percentage,
                                                                           weights_columns = {'n_seizures': 0.4,
                                                                                              'hours_of_data': 0.4},
                                                                           seed=SEED)
    train_subjects, validation_subjects, test_subjects = next(CV_generator)
    # recs = get_recs_list(config.data_path, config.locations, train_subjects)
    recs = get_recs_list(config.data_path, config.locations, validation_subjects)
    needed_recs = [# ['University_Hospital_Leuven_Pediatric', 'SUBJ-1b-315', 'r4'],
            ['University_Hospital_Leuven_Adult', 'SUBJ-1a-006', 'r10'],
            ]
    needed_recs_indices = [recs.index(r) for r in needed_recs if r in recs]
    # path_segments = get_paths_segments_train(config, config.get_name(), 0)
    path_segments = os.path.join("..", get_paths_segments_val(config, config.get_name(), 0))
    if os.path.exists(path_segments):
        with open(path_segments, 'rb') as inp:
            segments = pickle.load(inp)
            print("There are ", len(segments), "segments")

    # faulty_segment = segments[1]
    # print("Look at this segment:", faulty_segment, "of recording", recs[int(faulty_segment[0])])
    # train_segments = generate_data_keys_subsample(config, train_recs_list)
    ####################################################
    # Generate TFRecord paths for each segment
    tfrecord_files = []
    for i, s in tqdm(enumerate(segments), desc="Preparing TFRecord files"):
        rec_idx, start, stop, _ = s
        if int(rec_idx) not in needed_recs_indices:
            continue
        path = get_path_tfrecord(config.data_path, recs[int(rec_idx)], start, stop)
        tfrecord_files.append(path)
        if not os.path.exists(path):
            create_single_tfrecord(config, recs, s)

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    def parse_test(example_proto):
        features = {
            "segment": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([2], tf.float32),
        }
        return tf.io.parse_single_example(example_proto, features)
    dataset = dataset.map(parse_test)

    gen = dataset.batch(1)
    ####################################################
    # gen_train, _ = build_tfrecord_dataset(config, train_recs_list, train_segments, batch_size=config.batch_size,
    #                                    shuffle=False)
    segment_shape = (config.frame * config.fs, config.CH, 1)
    for i, raw_segment in enumerate(gen):
        segment_data = tf.io.decode_raw(raw_segment["segment"], tf.float32)
        if segment_data.shape[1] != 10500:
            print(i)
            print("Segment", segments[i], "of recording", recs[int(segments[i][0])])
            print("Segment data shape before reshape:", segment_data.shape)
        # print("Desired segment shape:", segment_shape)
        # segment_data = tf.reshape(segment_data, segment_shape)
        # if segment[0].shape[-2] != 21:
        #     print("Segment", segments[i], "of recording", recs[int(segments[i][0])],
        #           "has shape", segment[0].shape, "instead of (?, 21)")
        #     continue
        # rec_index = int(segments[i][0])
        # print("Segment", segments[i], "of recording", recs[rec_index], ": loading ok")
    #
    # val_recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)
    #
    # path_segments_val = get_paths_segments_val(config, config.get_name(), 0)
    # if os.path.exists(path_segments_val):
    #     with open(get_paths_segments_val(config, config.get_name(), 0), 'rb') as inp:
    #         val_segments = pickle.load(inp)
    #         print("There are ", len(val_segments), "segments in the validation set")
    # # val_segments = generate_data_keys_sequential_window(config, val_recs_list, config.val_batch_size)
    # gen_val, _ = build_tfrecord_dataset(config, val_recs_list, val_segments, batch_size=config.val_batch_size,
    #                                  shuffle=False)
    #
    # for i, segment in enumerate(gen_val):
    #     if segment[0].shape[-2] != 21:
    #         print("Segment", val_segments[i], "of recording", val_recs_list[int(val_segments[i][0])],
    #               "has shape", segment[0].shape, "instead of (?, 21)")
    #         continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    inspect_channels()
    # negative_dimensions()
    # ts_reshape_error()