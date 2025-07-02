import math
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from utility.constants import SEED, subjects_with_seizures, excluded_subjects, Locations


def leave_one_person_out(root_dir: str, included_locations: list[str] = None, validation_set: Optional[float] = None,
                         seed: int = SEED):
    testing = 'dtai' not in root_dir
    print("Testing setting:", testing)
    if testing:
        nb_subjects = 3 # ONLY FOR TESTING
        if Locations.leuven_adult in included_locations:
            included_subjects = ['SUBJ-1a-159', 'SUBJ-1a-358', 'SUBJ-1a-153']  # Leuven Adult subjects
        else:
            included_subjects = ['SUBJ-7-331', 'SUBJ-7-379', 'SUBJ-7-376'] # Coimbra subjects
    else:
        nb_subjects = None
        included_subjects = None
    all_subjects = []
    for location in os.listdir(root_dir):
        if included_locations is not None and location not in included_locations:
            continue
        location_path = os.path.join(root_dir, location)
        if os.path.isdir(location_path):  # Ensure it's a folder
            for subject in os.listdir(location_path):
                # REMOVE IF NOT TESTING
                if testing and included_subjects is not None and subject not in included_subjects:
                    continue
                if subject in excluded_subjects:
                    continue
                all_subjects.append(subject)
                # REMOVE IF NOT TESTING
                if testing and len(all_subjects) == nb_subjects:
                    break

    for subject in all_subjects:
        seed += 1
        train = all_subjects.copy()
        train.remove(subject)
        if validation_set is not None:
            n = math.ceil(validation_set * len(train))
            valid_validation_set = False
            while not valid_validation_set:
                rng = np.random.default_rng(seed)
                rng.shuffle(train)
                # Brute force to ensure at least one person with a seizure in the validation set
                # TODO: make this nicer
                if len(set(train[:n]).intersection(set(subjects_with_seizures))) > 0:
                    valid_validation_set = True
                else:
                    seed += 1
            yield train[n:], train[:n], [subject]
        else:
            yield train, [subject]

def multi_objective_grouped_stratified_cross_validation(info_per_group: pd.DataFrame, group_column: str,
                                                        id_column: str, n_splits: int,
                                                        train_size: float, val_size: float, seed=SEED):
    np.random.seed(seed)
    df = info_per_group.copy()
    assert train_size + val_size < 1, ("Train and validation sizes must sum to less than 1. The rest will be used for "
                                       "the test set.")
    assert n_splits > 1, "n_splits must be greater than 1 for cross-validation."
    test_size = 1 - train_size - val_size

    group_names = df[group_column].unique()
    groups = {name: df[df[group_column] == name] for name in group_names}
    totals_per_group = pd.DataFrame({name: group[[c for c in group.columns if c != id_column]].sum(axis=0) for name, group in groups.items()})
    split_targets = {'train': {name: train_size * totals for name, totals in totals_per_group.items()},
                     'val': {name: val_size * totals for name, totals in totals_per_group.items()},
                     'test': {name: test_size * totals for name, totals in totals_per_group.items()}}

    for split in range(n_splits):
        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=seed + split).reset_index(drop=True)

        # Initialize tracking structures
        assignments = defaultdict(list)
        current_sums = {
            'train': defaultdict(float),
            'val': defaultdict(float),
            'test': defaultdict(float)
        }

        for _, row in df.iterrows():
            pid = row[id_column]

        # # Initialize counters for each split
        # split_counts = {key: {col: 0 for col in df.columns if col != id_column} for key in split_targets.keys()}
        #
        # train_indices, val_indices, test_indices = [], [], []
        #
        # for idx, row in df.iterrows():
        #     group_id = row[id_column]
        #     group_data = row.drop(id_column)
        #
        #     # Determine which split this group should go into
        #     for split_name, target in split_targets.items():
        #         if all(split_counts[split_name][col] + group_data[col] <= target[col] for col in group_data.index):
        #             if split_name == 'train':
        #                 train_indices.append(group_id)
        #             elif split_name == 'val':
        #                 val_indices.append(group_id)
        #             elif split_name == 'test':
        #                 test_indices.append(group_id)
        #
        #             # Update the counts
        #             for col in group_data.index:
        #                 split_counts[split_name][col] += group_data[col]
        #             break
        #
        # yield train_indices, val_indices, test_indices

# def leave_one_seizure_out(X: pl.DataFrame, y: pl.Series):
#     end_previous_seizure = None
#     previous_fold_split = 0
#     in_seizure = False
#     for i in range(len(y)):
#         if not in_seizure and y[i] == 1:
#             in_seizure = True
#             if end_previous_seizure is None:
#                 continue
#             start_next_seizure = i
#             new_fold_split = (end_previous_seizure + start_next_seizure) // 2
#             X_train = pl.concat([X.slice(0, previous_fold_split), X.slice(new_fold_split, len(X) - new_fold_split)])
#             y_train = pl.concat([y.slice(0, previous_fold_split), y.slice(new_fold_split, len(y) - new_fold_split)])
#             X_test = X.slice(previous_fold_split, new_fold_split - previous_fold_split)
#             y_test = y.slice(previous_fold_split, new_fold_split - previous_fold_split)
#             previous_fold_split = new_fold_split
#
#             yield X_train, y_train, X_test, y_test
#
#         elif in_seizure and y[i] == 0:
#             in_seizure = False
#             end_previous_seizure = i

