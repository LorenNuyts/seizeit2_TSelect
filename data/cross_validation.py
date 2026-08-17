import math
import os
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd

from analysis.dataset import dataset_stats
from utility.constants import SEED, subjects_Fz_reference, excluded_subjects, Keys, subjects_with_seizures
from utility.debug_settings import debug_pool


def leave_one_person_out(root_dir: str, included_locations: list[str] = None, validation_set: Optional[float] = None,
                         seed: int = SEED, debug_subjects: Optional[List[str]] = None, n_debug_subjects: int = 0):
    all_subjects = []
    for location in os.listdir(root_dir):
        if included_locations is not None and location not in included_locations:
            continue
        location_path = os.path.join(root_dir, location)
        if os.path.isdir(location_path):  # Ensure it's a folder
            for subject in os.listdir(location_path):
                if subject in excluded_subjects:
                    continue
                all_subjects.append(subject)

    if debug_subjects:  # A debug run: only a handful of subjects, see utility/debug_settings.py
        all_subjects = debug_pool(debug_subjects, n_debug_subjects, all_subjects, minimum=2)

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
                             seed: int = SEED, debug_subjects: Optional[List[str]] = None, n_debug_subjects: int = 0):
    if debug_subjects:
        # A debug run (see utility/debug_settings.py): a handful of subjects, so that the group that
        # is left out is a handful of subjects too instead of a whole group. Groups without any of
        # those subjects are dropped, so a debug run has fewer folds than a full run.
        pool = debug_pool(debug_subjects, n_debug_subjects, info_per_group[id_column].unique().tolist(),
                          minimum=2 if validation_set is None else 3)
        info_per_group = info_per_group[info_per_group[id_column].isin(pool)]

    all_groups = info_per_group[group_column].unique()
    for i, group in enumerate(all_groups):
        np.random.seed(seed + i)
        test_ids = info_per_group[info_per_group[group_column] == group][id_column].unique().tolist()
        train_groups = [g for g in all_groups if g != group]
        if validation_set is not None:
            train_val_df = info_per_group[info_per_group[group_column].isin(train_groups)]
            if debug_subjects and train_val_df[id_column].nunique() < 2:
                # A debug run does not have enough subjects left to both train and validate on
                print("Debug run: skipping the fold that leaves out {}, since only {} subjects "
                      "would be left to train and validate on.".format(group, train_val_df[id_column].nunique()))
                continue
            train_ids, val_ids = next(multi_objective_grouped_stratified_cross_validation(train_val_df,
                                                                                     group_column=group_column,
                                                                                     id_column=id_column,
                                                                                     n_splits=1,
                                                                                     subset_sizes=[1 - validation_set, validation_set],
                                                                                     weights_columns={'n_seizures': 0.4,
                                                                                                      'hours_of_data': 0.4},
                                                                                     seed=seed + i,
                                                                                     debug_subjects=debug_subjects,
                                                                                     n_debug_subjects=n_debug_subjects))
            yield train_ids, val_ids, test_ids
        else:
            train_ids = info_per_group[info_per_group[group_column].isin(train_groups)][id_column].unique().tolist()
            yield train_ids, test_ids


def _numeric_metrics(df: pd.DataFrame, id_column: str, group_column: str) -> List[str]:
    """ The numeric columns of the dataframe that are stratified over, i.e. every column that is not
    the ID or the group column. """
    return [c for c in df.columns
            if c not in (id_column, group_column) and pd.api.types.is_numeric_dtype(df[c])]


def _stratification_weights(extended_metrics: List[str], weights_columns: Optional[dict]) -> dict:
    """
    The weight of every quantity that is balanced. The weights sum to one: the quantities that are
    not given a weight share the weight that is left over. With
    {'n_seizures': 0.4, 'hours_of_data': 0.4} and the number of subjects as the only other quantity,
    this gives the weighting of the paper: seizures and hours weigh twice as heavily as the subject
    count.
    """
    if weights_columns is None:
        return {k: 1 / len(extended_metrics) for k in extended_metrics}

    unknown = set(weights_columns) - set(extended_metrics)
    assert not unknown, "Weights were given for columns that are not stratified over: {}".format(sorted(unknown))
    total_weight = sum(weights_columns.values())
    assert 0 <= total_weight <= 1, "The given weights must sum to at most 1, but they sum to {}".format(total_weight)
    missing_weights = [k for k in extended_metrics if k not in weights_columns]
    assert missing_weights or math.isclose(total_weight, 1, abs_tol=1e-6), \
        "The given weights cover every column but do not sum to 1 (they sum to {})".format(total_weight)
    rest = (1 - total_weight) / len(missing_weights) if missing_weights else 0
    return {k: weights_columns[k] if k in weights_columns else rest for k in extended_metrics}


def _blocks_per_subset(subset_sizes: List[float], max_blocks: int = 200, tolerance: float = 1e-4):
    """
    Translates the requested subset sizes into a number of blocks to partition the subjects into and
    a number of those blocks per subset. The smallest number of blocks is taken for which every
    subset gets a whole number of blocks that matches its requested size, e.g. [0.8, 0.1, 0.1] gives
    10 blocks divided as 8/1/1.

    Returns the number of blocks and the number of blocks of every subset.
    """
    assert math.isclose(sum(subset_sizes), 1, abs_tol=1e-6), \
        "The sum of subset sizes must be 1, but got: {}".format(sum(subset_sizes))
    assert all(size > 0 for size in subset_sizes), \
        "Every subset size must be positive, but got: {}".format(subset_sizes)

    for n_blocks in range(len(subset_sizes), max_blocks + 1):
        counts = [int(round(size * n_blocks)) for size in subset_sizes]
        if sum(counts) != n_blocks or min(counts) < 1:
            continue
        if max(abs(count / n_blocks - size) for count, size in zip(counts, subset_sizes)) <= tolerance:
            return n_blocks, counts

    raise ValueError("Could not express the subset sizes {} as whole blocks of at most {} blocks. Use subset sizes "
                     "that are multiples of a common fraction, e.g. [0.8, 0.1, 0.1].".format(subset_sizes, max_blocks))


def _tied_at_minimum(values: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """
    The indices of the values that are at the minimum, up to a relative tolerance. The tolerance
    keeps the result independent of the order in which the totals behind these values happened to be
    summed, which otherwise makes the blocks depend on the row order of the dataframe.
    """
    minimum = values.min()
    return np.flatnonzero(values <= minimum + tolerance * max(1.0, abs(minimum)))


def _greedy_grouped_stratified_partition(df: pd.DataFrame, group_column: str, id_column: str, n_blocks: int,
                                         weights_columns: Optional[dict], seed: int) -> List[List]:
    """
    Partitions the IDs of the dataframe into n_blocks disjoint blocks of roughly equal size.

    Within each group (e.g. each hospital), every block should receive a share of each metric (e.g.
    the seizures, the hours of recording) and of the IDs proportional to its size, i.e. 1/n_blocks
    of the totals of that group. IDs are indivisible and are assigned one at a time: each ID is
    placed in the block whose shares are currently furthest from these proportions. Concretely, the
    ID is placed in the block for which the total imbalance

        sum over blocks b, groups g and metrics k of  weight[k] * |target[b, g, k] - current[b, g, k]| / target[b, g, k]

    increases the least. Since adding an ID only changes the block it is added to and only for its
    own group, this is the block with the smallest increase of its own group's imbalance.

    Two details to make sure the blocks are balanced:
      * The IDs are assigned largest first: in decreasing order of how much of its group's totals an
        ID holds. The seizures are extremely lumpy (one subject holds 55 of the 91 seizures of Leuven
        Adult), and an ID that large can only be compensated if it is placed while the blocks are
        still empty. The order within equally large IDs is random but seeded.
      * Ties are broken by the block that is furthest from its share of the totals over *all* groups.
        Without this, the largest ID of every group goes to the first block (all blocks are equally
        empty and equally good for its own group), so the large IDs of the different groups pile up
        in the same blocks.
    The remainder is irreducible: no partition can split a subject holding 55 seizures.

    Returns the IDs of every block.
    """
    metrics = _numeric_metrics(df, id_column, group_column)
    assert metrics, "There are no numeric columns to stratify over in {}".format(list(df.columns))
    extended_metrics = metrics + ['n_ids']  # Also balance the number of IDs in each block
    weights = _stratification_weights(extended_metrics, weights_columns)
    weight_vector = np.array([weights[k] for k in extended_metrics], dtype=np.float64)

    group_names = list(df[group_column].unique())
    group_indices = {name: i for i, name in enumerate(group_names)}

    totals = df.groupby(group_column)[metrics].sum()
    totals['n_ids'] = df.groupby(group_column)[id_column].nunique()
    totals = totals.loc[group_names, extended_metrics]
    # Each block should receive an equal share of the totals of each group ...
    targets = totals.to_numpy(dtype=np.float64) / n_blocks  # (n_groups, n_extended_metrics)
    # ... and, only to break ties, an equal share of the totals over all groups
    overall_targets = totals.to_numpy(dtype=np.float64).sum(axis=0) / n_blocks
    # A group without any of a metric (e.g. a hospital without seizures) cannot be imbalanced in it
    balanced = targets > 0
    overall_balanced = overall_targets > 0

    # Shuffle first so that IDs holding an equally large share are assigned in a random, seeded
    # order. Sort by ID first, so that the blocks depend only on the IDs, their metrics and the
    # seed, and not on the order in which the rows happen to arrive.
    shuffled = df.sort_values(id_column).sample(frac=1, random_state=seed)
    shares = np.zeros(len(shuffled), dtype=np.float64)
    for k in metrics:
        group_totals = shuffled[group_column].map(totals[k]).to_numpy(dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            shares += weights[k] * np.where(group_totals > 0,
                                            shuffled[k].to_numpy(dtype=np.float64) / group_totals, 0)
    shuffled = shuffled.iloc[np.argsort(-shares, kind='stable')]  # largest first

    blocks = [[] for _ in range(n_blocks)]
    current = np.zeros((n_blocks, len(group_names), len(extended_metrics)), dtype=np.float64)

    for _, row in shuffled.iterrows():
        group_index = group_indices[row[group_column]]
        contribution = np.array([1.0 if k == 'n_ids' else float(row[k]) for k in extended_metrics])

        def imbalance_increase(before, target, in_use):
            """ The increase of the imbalance for every candidate block. """
            increase = np.zeros_like(before)
            increase[:, in_use] = ((np.abs(target[in_use] - (before[:, in_use] + contribution[in_use]))
                                    - np.abs(target[in_use] - before[:, in_use])) / target[in_use])
            return increase.dot(weight_vector)

        # (n_blocks,): the increase of the imbalance of the group of this ID per candidate block
        increase = imbalance_increase(current[:, group_index, :], targets[group_index], balanced[group_index])
        tied = _tied_at_minimum(increase)
        if len(tied) > 1:
            overall_increase = imbalance_increase(current.sum(axis=1), overall_targets, overall_balanced)
            tied = tied[_tied_at_minimum(overall_increase[tied])]
        best_block = int(tied[0])  # Any ties that are left are broken by the index of the block

        blocks[best_block].append(row[id_column])
        current[best_block, group_index, :] += contribution

    return blocks


def _debug_subset_sizes(n_ids: int, subset_sizes: List[float]) -> List[int]:
    """ Divides n_ids IDs over the subsets, giving every subset at least one ID. Only used by the
    hard-coded debug split of the testing setting. """
    assert n_ids >= len(subset_sizes), \
        "Cannot divide {} subjects over {} subsets".format(n_ids, len(subset_sizes))
    counts = [1] * len(subset_sizes)
    for i in sorted(range(len(subset_sizes)), key=lambda i: subset_sizes[i], reverse=True):
        if sum(counts) == n_ids:
            break
        counts[i] += n_ids - sum(counts)
    return counts


def multi_objective_grouped_stratified_cross_validation(info_per_group: pd.DataFrame, group_column: str,
                                                        id_column: str, n_splits: int, subset_sizes: List[float],
                                                        weights_columns: dict=None, seed=SEED,
                                                        debug_subjects: Optional[List[str]] = None,
                                                        n_debug_subjects: int = 0):
    """
    Grouped, stratified k-fold cross-validation.

    The IDs (subjects) are partitioned once into k disjoint blocks, where k follows from the
    requested subset sizes: [0.8, 0.1, 0.1] gives ten blocks of which eight form the training set,
    one the validation set and one the test set. The partition is made at the level of IDs so that
    all recordings of a subject end up in the same subset. It is stratified with the greedy
    multi-objective procedure of _greedy_grouped_stratified_partition: within each group (hospital),
    each block receives a share of the seizures, the hours of recording and the subjects
    proportional to its size, where seizures and hours weigh twice as heavily as the subject count
    when weights_columns={'n_seizures': 0.4, 'hours_of_data': 0.4} is given.

    Every split rotates the blocks over the subsets, so the test blocks of the splits are disjoint
    and every subject is tested on exactly once when n_splits equals the number of blocks. Since
    the blocks are only rotated, the number of subjects per subset is the same in every split
    and the subjects covered by a split are the same in every split. Note that with subset
    sizes [0.8, 0.1, 0.1] the validation set of a split is the test set of the previous split,
    which is inherent to rotating ten blocks over 8/1/1 subsets.

    The order in which the subjects are assigned and the way ties are broken are chosen to keep
    the blocks balanced; see _greedy_grouped_stratified_partition.

    :param info_per_group: one row per ID with the group column, the ID column and one numeric
        column per metric to stratify over (e.g. 'n_seizures' and 'hours_of_data').
    :param group_column: the column to stratify within, e.g. 'hospital'.
    :param id_column: the column with the IDs that are divided over the subsets, e.g. 'subject'.
    :param n_splits: the number of splits to yield. Splits repeat once it exceeds the number of blocks.
    :param subset_sizes: the requested proportion of each subset, summing to 1, e.g.
        [train, validation, test] = [0.8, 0.1, 0.1].
    :param weights_columns: the weight of each metric in the imbalance. The metrics without a weight,
        including the number of IDs, share the remaining weight. Defaults to equal weights.
    :param seed: the seed of the random order in which the IDs are assigned.
    :param debug_subjects: when set, this is a debug run (see utility/debug_settings.py): the splits
        are a handful of these subjects divided over the subsets, one per subset, instead of the
        real folds. None (the default) gives the real folds.
    :param n_debug_subjects: how many subjects a debug run falls back to when none of the given
        debug subjects are in info_per_group.
    :return: a generator of tuples with one list of IDs per subset size.
    """
    df = info_per_group.copy()
    df = df[~df[id_column].isin(excluded_subjects)]
    df = df.drop_duplicates(subset=id_column)

    if debug_subjects:
        # A debug run (see utility/debug_settings.py): these are not the real folds, just a handful
        # of subjects divided over the subsets to run the pipeline on.
        pool = debug_pool(debug_subjects, n_debug_subjects, df[id_column].unique().tolist(),
                          minimum=len(subset_sizes))
        counts = _debug_subset_sizes(len(pool), subset_sizes)
        offsets = np.cumsum([0] + counts)
        for split in range(n_splits):
            rng = np.random.default_rng(seed + split)
            permutation = rng.permutation(pool).tolist()
            yield tuple(permutation[offsets[i]:offsets[i + 1]] for i in range(len(counts)))
        return

    n_blocks, blocks_per_subset = _blocks_per_subset(subset_sizes)
    n_ids = df[id_column].nunique()
    assert n_ids >= n_blocks, \
        "Cannot divide {} {}s over {} blocks for subset sizes {}".format(n_ids, id_column, n_blocks, subset_sizes)
    if n_splits > n_blocks:
        print("Warning: {} splits were requested while the subset sizes {} only give {} blocks, so the splits "
              "repeat after {} splits.".format(n_splits, subset_sizes, n_blocks, n_blocks))

    blocks = _greedy_grouped_stratified_partition(df, group_column, id_column, n_blocks, weights_columns, seed)

    # Rotate the blocks over the subsets by the size of the last subset (the test set), so that
    # consecutive splits have disjoint test sets.
    shift = blocks_per_subset[-1]
    offsets = np.cumsum([0] + blocks_per_subset)
    for split in range(n_splits):
        order = [(block + split * shift) % n_blocks for block in range(n_blocks)]
        yield tuple([pid for block in order[offsets[i]:offsets[i + 1]] for pid in blocks[block]]
                    for i in range(len(blocks_per_subset)))


def multi_objective_grouped_stratified_random_sampling(info_per_group: pd.DataFrame, group_column: str,
                                                        id_column: str, n_splits: int, subset_sizes: List[float],
                                                        weights_columns: dict=None, seed=SEED,
                                                        debug_subjects: Optional[List[str]] = None,
                                                        n_debug_subjects: int = 0):
    np.random.seed(seed)
    df = info_per_group.copy()

    # Remove the excluded subjects
    df = df[~df[id_column].isin(excluded_subjects)]

    if debug_subjects:
        # A debug run (see utility/debug_settings.py): a handful of subjects, one per subset
        pool = debug_pool(debug_subjects, n_debug_subjects, df[id_column].unique().tolist(),
                          minimum=len(subset_sizes))
        counts = _debug_subset_sizes(len(pool), subset_sizes)
        offsets = np.cumsum([0] + counts)
        for split in range(n_splits):
            rng = np.random.default_rng(seed + split)
            permutation = rng.permutation(pool).tolist()
            yield tuple(permutation[offsets[i]:offsets[i + 1]] for i in range(len(counts)))
        return

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
    # A debug run is restricted to a handful of subjects; a full run leaves these at None/0. The
    # setting is resolved by the entry point, see utility/debug_settings.py.
    debug_subjects = getattr(config, 'debug_subjects', None)
    n_debug_subjects = getattr(config, 'n_debug_subjects', 0)

    if config.cross_validation == Keys.leave_one_person_out:
        raise NotImplementedError("Leave one person out cross-validation is outdated.")
        if config.held_out_fold:
            raise NotImplementedError("Leave one person out cross-validation with held out fold is not implemented yet.")
        CV_generator = leave_one_person_out(config.data_path, included_locations=config.locations,
                                            validation_set=config.validation_percentage,
                                            debug_subjects=debug_subjects, n_debug_subjects=n_debug_subjects)

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
                                                                    seed=SEED,
                                                                    debug_subjects=debug_subjects,
                                                                    n_debug_subjects=n_debug_subjects)
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
                                                                           seed=SEED,
                                                                           debug_subjects=debug_subjects,
                                                                           n_debug_subjects=n_debug_subjects)
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
                                           validation_set=config.validation_percentage, seed=SEED,
                                           debug_subjects=debug_subjects, n_debug_subjects=n_debug_subjects)

    else:
        raise NotImplementedError('Cross-validation method not implemented yet')
    return CV_generator, held_out_subjects
