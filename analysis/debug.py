import argparse
import os
import pickle

from analysis.dataset import dataset_stats
from data.cross_validation import multi_objective_grouped_stratified_cross_validation
from net.DL_config import get_base_config
from net.generator_ds import build_tfrecord_dataset
from net.key_generator import generate_data_keys_sequential_window, generate_data_keys_sequential, \
    generate_data_keys_subsample
from utility import get_recs_list
from utility.constants import Locations, SEED
from utility.paths import get_paths_segments_train, get_paths_segments_val

base_ = os.path.dirname(os.path.realpath(__file__))

def negative_dimensions():
    config = get_base_config(os.path.join(base_, ".."), locations=[Locations.karolinska, Locations.freiburg])
    val_recs_list = [#['Karolinska_Institute', 'SUBJ-6-430', 'r26'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r1'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r2'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r3'],
                     ]
    val_segments = generate_data_keys_sequential(config, val_recs_list, 6 * 60)

def ts_reshape_error():
    config = get_base_config(os.path.join(base_, ".."), locations=sorted([Locations.karolinska, Locations.freiburg,
                                                                          Locations.leuven_adult, Locations.leuven_pediatric,
                                                                          Locations.aachen, Locations.coimbra]))
    info_per_group = dataset_stats(config.data_path, os.path.join(config.save_dir, "dataset_stats"), config.locations)

    CV_generator = multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                           id_column='subject',
                                                                           n_splits=config.n_folds,
                                                                           train_size=config.train_percentage,
                                                                           val_size=config.validation_percentage,
                                                                           weights_columns = {'n_seizures': 0.4,
                                                                                              'hours_of_data': 0.4},
                                                                           seed=SEED)
    train_subjects, validation_subjects, test_subjects = next(CV_generator)
    train_recs_list = get_recs_list(config.data_path, config.locations, train_subjects)
    path_segments_train = get_paths_segments_train(config, config.get_name(), 0)
    if os.path.exists(path_segments_train):
        with open(path_segments_train, 'rb') as inp:
            train_segments = pickle.load(inp)
            print("There are ", len(train_segments), "segments in the training set")

    path_segments_val = get_paths_segments_val(config, config.get_name(), 0)
    if os.path.exists(path_segments_val):
        with open(get_paths_segments_val(config, config.get_name(), 0), 'rb') as inp:
            val_segments = pickle.load(inp)
            print("There are ", len(val_segments), "segments in the validation set")
    # train_segments = generate_data_keys_subsample(config, train_recs_list)
    gen_train = build_tfrecord_dataset(config, train_recs_list, train_segments, batch_size=config.batch_size,
                                       shuffle=True)
    for i, segment in enumerate(gen_train):
        rec_index = int(train_segments[i][0])
        print("Segment", train_segments[i], "of recording", train_recs_list[rec_index], ": loading ok")

    val_recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)
    # val_segments = generate_data_keys_sequential_window(config, val_recs_list, config.val_batch_size)
    gen_val = build_tfrecord_dataset(config, val_recs_list, val_segments, batch_size=config.val_batch_size,
                                     shuffle=False)

    for i, segment in enumerate(gen_val):
        rec_index = int(val_segments[i][0])
        print("Segment", val_segments[i], "of recording", val_recs_list[rec_index], ": loading ok")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # negative_dimensions()
    ts_reshape_error()