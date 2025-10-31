import math
import os
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd

from analysis.dataset import dataset_stats
from utility.constants import SEED, subjects_Fz_reference, excluded_subjects, Locations, Keys, subjects_with_seizures


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

def leave_one_group_out(info_per_group: pd.DataFrame, group_column: str, id_column: str, validation_set: Optional[float] = None,
                             seed: int = SEED):
    all_groups = info_per_group[group_column].unique()
    for i, group in enumerate(all_groups):
        np.random.seed(seed + i)
        test_ids = info_per_group[info_per_group[group_column] == group][id_column].unique().tolist()
        train_groups = [g for g in all_groups if g != group]
        if validation_set is not None:
            train_val_df = info_per_group[info_per_group[group_column].isin(train_groups)]
            train_ids, val_ids = next(multi_objective_grouped_stratified_cross_validation(train_val_df,
                                                                                     group_column=group_column,
                                                                                     id_column=id_column,
                                                                                     n_splits=1,
                                                                                     subset_sizes=[1 - validation_set, validation_set],
                                                                                     weights_columns={'n_seizures': 0.4,
                                                                                                      'hours_of_data': 0.4},
                                                                                     seed=seed + i))
            yield train_ids, val_ids, test_ids
        else:
            train_ids = info_per_group[info_per_group[group_column].isin(train_groups)][id_column].unique().tolist()
            yield train_ids, test_ids

def multi_objective_grouped_stratified_cross_validation(info_per_group: pd.DataFrame, group_column: str,
                                                        id_column: str, n_splits: int, subset_sizes: List[float],
                                                        weights_columns: dict=None, seed=SEED):
    np.random.seed(seed)
    testing = 'dtai' not in os.path.dirname(os.path.realpath(__file__))
    print("Testing setting:", testing)
    if testing:
        locations = info_per_group['hospital'].unique()
        if Locations.leuven_adult in locations:
            included_subjects = ['SUBJ-1a-159', 'SUBJ-1a-358', 'SUBJ-1a-153']  # Leuven Adult subjects
        else:
            included_subjects = ['SUBJ-7-331', 'SUBJ-7-379', 'SUBJ-7-376']  # Coimbra subjects
        # yield permutations of these subjects
        for i in range(n_splits):
            np.random.seed(seed + i)
            perm = np.random.permutation(included_subjects).tolist()
            yield perm[0], perm[1], perm[2]
        return
    df = info_per_group.copy()

    # Remove the excluded subjects
    df = df[~df[id_column].isin(excluded_subjects)]

    assert sum(subset_sizes) == 1, ("The sum of subset sizes must be 1, but got: {}".format(sum(subset_sizes)))

    group_names = df[group_column].unique()
    groups = {name: df[df[group_column] == name] for name in group_names}
    metrics = [c for c in df.columns if c not in [id_column, group_column]]
    totals_per_group = pd.DataFrame({name: group[[c for c in metrics]].sum(axis=0) for name, group in groups.items()}).transpose()
    totals_per_group['n_ids'] = df.groupby(group_column)[id_column].nunique()
    split_targets = {i: size * totals_per_group for i, size in enumerate(subset_sizes)}
    totals = totals_per_group.sum(axis=0)
    totals['n_ids'] = df.shape[0]

    extended_metrics = metrics + ['n_ids']  # Add a column to track the number of IDs in each group
    if weights_columns is None:
        weights = {k: 1/(len(extended_metrics)) for k in extended_metrics}
    else:
        total_weight = sum(weights_columns.values())
        missing_weights = set(extended_metrics) - set(weights_columns.keys())
        weights = {k: weights_columns[k] if k in weights_columns.keys() else (1-total_weight)/len(missing_weights) for k in extended_metrics}

    folds = list(split_targets.keys())
    for split in range(n_splits):
        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=seed + split).reset_index(drop=True)

        # Initialize tracking structures
        assignments = {fold: [] for fold in folds}  # Assignments for each fold
        current_sums = {fold:  pd.DataFrame(0, index=group_names, columns=extended_metrics, dtype=np.float64) for fold in folds}

        imbalance_per_fold = [(len(group_names)) for _ in folds]  # Initialize with the number of groups for each fold. Each group can have at most an imbalance of 1.
        for _, row in df.iterrows():
            pid = row[id_column]
            current_group = row[group_column]
            current_metrics = row[metrics].to_dict()
            current_metrics['n_ids'] = 1

            candidate_total_imbalances = []
            new_imbalance_per_fold = []
            for i, fold in enumerate(folds):
                new_imbalance_per_fold.append(0)
                for k in extended_metrics:
                    target = split_targets[fold].loc[current_group][k]
                    current = current_sums[fold].loc[current_group][k]
                    new_current = current + current_metrics[k]
                    new_imbalance_per_fold[i] += abs(target - new_current) / target * weights[k]

                    # Add metric for the other groups
                    for group in group_names:
                        if group != current_group:
                            target_other = split_targets[fold].loc[group][k]
                            current_other = current_sums[fold].loc[group][k]
                            new_imbalance_per_fold[i] += abs(target_other - current_other) / target_other * weights[k]

                # The candidate total imbalance is the new imbalance for this fold plus the previous imbalances of the other folds
                candidate_total_imbalances.append(new_imbalance_per_fold[i] + sum([imb for j, imb in enumerate(imbalance_per_fold) if i != j]))

            min_idx = candidate_total_imbalances.index(min(candidate_total_imbalances))
            best_fold = folds[min_idx]

            imbalance_per_fold[min_idx] = new_imbalance_per_fold[min_idx]  # Update the previous imbalance for the best fold

            assignments[best_fold].append(pid)
            for k in extended_metrics:
                current_sums[best_fold].loc[current_group, k] += current_metrics[k]

        yield tuple([v for v in assignments.values()])


def get_CV_generator(config):
    held_out_subjects = []

    if config.cross_validation == Keys.leave_one_person_out:
        raise NotImplementedError("Leave one person out cross-validation is outdated.")
        if config.held_out_fold:
            raise NotImplementedError("Leave one person out cross-validation with held out fold is not implemented yet.")
        CV_generator = leave_one_person_out(config.data_path, included_locations=config.locations,
                                            validation_set=config.validation_percentage)

    elif config.cross_validation == Keys.stratified:
        info_per_group = dataset_stats(config.data_path, os.path.join(config.save_dir, "dataset_stats"),
                                       config.locations)
        test_size = 1 - (config.train_percentage + config.validation_percentage)
        if config.Fz_reference:
            info_per_group = info_per_group[info_per_group['subject'].isin(subjects_Fz_reference)]
        else:
            info_per_group = info_per_group[~info_per_group['subject'].isin(subjects_Fz_reference)]
        if config.held_out_fold:
            gen = multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                    id_column='subject',
                                                                    n_splits=1,
                                                                    subset_sizes=[1 - test_size, test_size]
                                                                    , weights_columns={'n_seizures': 0.4,
                                                                                        'hours_of_data': 0.4},
                                                                    seed=SEED)
            other_subjects, held_out_subjects = next(gen)
            info_per_group = info_per_group[info_per_group['subject'].isin(other_subjects)]
        CV_generator = multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                           id_column='subject',
                                                                           n_splits=config.n_folds,
                                                                            subset_sizes=[config.train_percentage,
                                                                                          config.validation_percentage,
                                                                                          test_size],
                                                                           weights_columns={'n_seizures': 0.4,
                                                                                            'hours_of_data': 0.4},
                                                                           seed=SEED)
    elif config.cross_validation == Keys.leave_one_hospital_out:
        info_per_group = dataset_stats(config.data_path, os.path.join(config.save_dir, "dataset_stats"),
                                       config.locations)
        if config.Fz_reference:
            info_per_group = info_per_group[info_per_group['subject'].isin(subjects_Fz_reference)]
        else:
            info_per_group = info_per_group[~info_per_group['subject'].isin(subjects_Fz_reference)]
        if config.held_out_fold:
            test_size = 1 - (config.train_percentage + config.validation_percentage)
            df = info_per_group.copy()
            group_column = 'hospital'
            id_column = 'subject'
            groups = {name: df[df['hospital'] == name] for name in config.locations}
            metrics = [c for c in df.columns if c not in [id_column, group_column]]
            totals_per_hospital = pd.DataFrame(
                {name: group[[c for c in metrics]].sum(axis=0) for name, group in groups.items()}).transpose()
            totals_per_hospital['n_ids'] = df.groupby(group_column)[id_column].nunique()
            totals = totals_per_hospital.sum(axis=0)
            split_targets: list = totals * test_size

            weights = {'n_seizures': 0.4, 'hours_of_data': 0.4, 'n_ids': 0.2}
            best_hospital = None
            best_score = float('inf')
            for hospital in config.locations:
                score = abs(split_targets - totals_per_hospital.loc[hospital]).dot(pd.Series(weights))
                if score < best_score:
                    best_score = score
                    best_hospital = hospital

            held_out_subjects = info_per_group[info_per_group['hospital'] == best_hospital]['subject'].unique().tolist()
            info_per_group = info_per_group[info_per_group['hospital'] != best_hospital]

        CV_generator = leave_one_group_out(info_per_group, group_column='hospital', id_column='subject',
                                           validation_set=config.validation_percentage, seed=SEED)

    else:
        raise NotImplementedError('Cross-validation method not implemented yet')
    return CV_generator, held_out_subjects
