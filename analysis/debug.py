import argparse
import copy
import os
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from analysis.channel_analysis.file_management import download_remote_configs
from analysis.dataset import dataset_stats
from data.cross_validation import multi_objective_grouped_stratified_cross_validation
from net.DL_config import get_base_config, get_channel_selection_config
from net.generator_ds import build_tfrecord_dataset, parse_example
from net.key_generator import generate_data_keys_sequential_window, generate_data_keys_sequential, \
    generate_data_keys_subsample
from utility import get_recs_list
from utility.constants import Locations, SEED, parse_location, Nodes, evaluation_metrics, Keys
from data.data import create_single_tfrecord, Data
from utility.paths import get_paths_segments_train, get_paths_segments_val, get_path_tfrecord, get_path_config

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
                     # ['University_Hospital_Leuven_Adult', 'SUBJ-1a-006', 'r10'],
        ['Freiburg_University_Medical_Center', 'SUBJ-4-381', 'r6'],
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

def changed_channels_rereferencing():
    config = get_base_config(os.path.join(base_, ".."), locations=[parse_location(l) for l in Locations.all_keys()])
    changed_channels = defaultdict(list)
    for location in config.locations:
        data_path = os.path.join(config.data_path, location)
        if not os.path.isdir(data_path):
            print("Data path", data_path, "is not a directory")
            continue

        for subject in os.listdir(data_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(data_path, subject)

            if not os.path.isdir(subject_path):
                print("     | Skipping subject because it is not a directory:", subject)
                continue

            recs = get_recs_list(config.data_path, [location], [subject])
            for recording in recs:
                print("         | Processing recording:", recording)
                rec_data = Data.loadData(config.data_path, recording,
                                         included_channels=config.included_channels, load_preprocessed=False)
                rec_data_rereferenced = copy.deepcopy(rec_data)
                rec_data.apply_preprocess(data_path=config.data_path, fs=config.fs, rereference_channel=False)
                rec_data_rereferenced.apply_preprocess(data_path=config.data_path, fs=config.fs, rereference_channel=True)

                for ch_index, ch in enumerate(config.included_channels):
                    data_rereferenced_ch = rec_data_rereferenced[ch_index]
                    data_ch = rec_data[ch_index]

                    if not np.array_equal(data_rereferenced_ch, data_ch):
                    # if not data_rereferenced_ch.equals(data_ch):
                        changed_channels[ch].append(subject)
                        if ch in Nodes.wearable_nodes:
                            print(f"         | Channel {ch} changed for subject {subject}")


def same_cross_validation_split():
    new_config = get_channel_selection_config(os.path.join(base_, ".."), locations=[parse_location(l) for l in Locations.all_keys()],
                                                              evaluation_metric=evaluation_metrics['score'],
                                                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                                              held_out_fold=True, pretty_name="Channel Selection")

    old_config = get_channel_selection_config(os.path.join(base_, ".."), locations=[parse_location(l) for l in Locations.all_keys()],
                                                              evaluation_metric=evaluation_metrics['score'],
                                                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                                              held_out_fold=True, pretty_name="Channel Selection",
                                                              version_experiments=None)
    for c in [new_config, old_config]:
        c_path = get_path_config(c, c.get_name())
        if os.path.exists(c_path):
            c.load_config(c_path, c.get_name())
        else:
            print(f"Config not found for {c.get_name()}, downloading...")
            download_remote_configs([c], local_base_dir=c.save_dir)
            c.load_config(c_path, c.get_name())

    new_folds = new_config.folds
    old_folds = old_config.folds
    for i in new_folds.keys():
        if i not in old_folds:
            print(f"Fold {i} is missing")

        for s in new_folds[i]:
            assert sorted(new_folds[i][s]) == sorted(old_folds[i][s])

    assert sorted(new_config.held_out_subjects) == sorted(old_config.held_out_subjects)
    print("All folds are identical.")

def same_segments():
    new_config = get_channel_selection_config(os.path.join(base_, ".."), locations=[parse_location(l) for l in Locations.all_keys()],
                                                              evaluation_metric=evaluation_metrics['score'],
                                                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                                              held_out_fold=True, pretty_name="Channel Selection")

    old_config = get_channel_selection_config(os.path.join(base_, ".."), locations=[parse_location(l) for l in Locations.all_keys()],
                                                              evaluation_metric=evaluation_metrics['score'],
                                                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                                              held_out_fold=True, pretty_name="Channel Selection",
                                                              version_experiments=None)
    for c in [new_config, old_config]:
        c_path = get_path_config(c, c.get_name())
        if os.path.exists(c_path):
            c.load_config(c_path, c.get_name())
        else:
            print(f"Config not found for {c.get_name()}, downloading...")
            download_remote_configs([c], local_base_dir=c.save_dir)
            c.load_config(c_path, c.get_name())

    folds = list(new_config.folds.keys())
    for fold_i in folds:
        new_path_segments_train = get_paths_segments_train(new_config, new_config.get_name(), fold_i)
        with open(new_path_segments_train, 'rb') as inp:
            new_train_segments: List[list] = pickle.load(inp)

        old_path_segments_train = get_paths_segments_train(old_config, old_config.get_name(), fold_i)
        with open(old_path_segments_train, 'rb') as inp:
            old_train_segments: List[list] = pickle.load(inp)

        print(f"Fold {fold_i}: New train segments: {len(new_train_segments)}, Old train segments: {len(old_train_segments)}")
        assert all(sorted(new_train_segments) == sorted(old_train_segments)), f"Train segments differ in fold {fold_i}"

        new_path_segments_val = get_paths_segments_val(new_config, new_config.get_name(), fold_i)
        with open(new_path_segments_val, 'rb') as inp:
            new_val_segments = pickle.load(inp)
        old_path_segments_val = get_paths_segments_val(old_config, old_config.get_name(), fold_i)
        with open(old_path_segments_val, 'rb') as inp:
            old_val_segments = pickle.load(inp)
        assert sorted(new_val_segments) == sorted(old_val_segments), f"Validation segments differ in fold {fold_i}"

    print("All segments in all folds are identical.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # inspect_channels()
    # negative_dimensions()
    # ts_reshape_error()
    # changed_channels_rereferencing()
    # same_cross_validation_split()
    same_segments()