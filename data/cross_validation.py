import math
import os
import warnings
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
                                                        train_size: float, val_size: float,
                                                        weights_columns: dict=None, seed=SEED):
    np.random.seed(seed)
    df = info_per_group.copy()

    # Remove the excluded subjects
    df = df[~df[id_column].isin(excluded_subjects)]
    
    assert train_size + val_size < 1, ("Train and validation sizes must sum to less than 1. The rest will be used for "
                                       "the test set.")
    assert n_splits > 1, "n_splits must be greater than 1 for cross-validation."
    test_size = 1 - train_size - val_size

    group_names = df[group_column].unique()
    groups = {name: df[df[group_column] == name] for name in group_names}
    metrics = [c for c in df.columns if c not in [id_column, group_column]]
    totals_per_group = pd.DataFrame({name: group[[c for c in metrics]].sum(axis=0) for name, group in groups.items()}).transpose()
    totals_per_group['n_ids'] = df.groupby(group_column)[id_column].nunique()
    split_targets = {'train': train_size * totals_per_group,
                     'val': val_size * totals_per_group,
                     'test': test_size * totals_per_group}
    totals = totals_per_group.sum(axis=0)
    totals['n_ids'] = df.shape[0]

    extended_metrics = metrics + ['n_ids']  # Add a column to track the number of IDs in each group
    if weights_columns is None:
        weights = {k: 1/(len(extended_metrics)) for k in extended_metrics}
    else:
        total_weight = sum(weights_columns.values())
        missing_weights = set(extended_metrics) - set(weights_columns.keys())
        weights = {k: weights_columns[k] if k in weights_columns.keys() else (1-total_weight)/len(missing_weights) for k in extended_metrics}

    folds = ['train', 'val', 'test']
    for split in range(n_splits):
        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=seed + split).reset_index(drop=True)

        # Initialize tracking structures
        assignments = defaultdict(list)
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

        yield assignments['train'], assignments['val'], assignments['test']
