import argparse
import os

import numpy as np

from analysis.dataset import dataset_stats
from data.cross_validation import multi_objective_grouped_stratified_cross_validation
from net.DL_config import get_base_config
from utility.constants import SEED, Keys, Locations

base_ = os.path.dirname(os.path.realpath(__file__))

def dissimilarity_across_data_splits(config):
    """
    splits: List of split dicts, where each dict has 'train', 'val', 'test' DataFrames with 'patient_id'
    Returns: dict of dissimilarity matrices (one per subset), each as a 2D numpy array (NxN)
    """
    info_per_group = dataset_stats(config.data_path, os.path.join("..", config.save_dir, "dataset_stats"), config.locations)
    splits = [x for x in multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                           id_column='subject',
                                                                           n_splits=config.n_folds,
                                                                           train_size=config.train_percentage,
                                                                           val_size=config.validation_percentage,
                                                                           weights_columns = {'n_seizures': 0.4,
                                                                                              'hours_of_data': 0.4},
                                                                           seed=SEED)]
    n = len(splits)
    n_folds = len(splits[0])
    matrices = {i: np.zeros((n, n)) for i in range(n_folds)}

    for f in range(n_folds):
        id_sets = [set(split[f]) for split in splits]

        for i in range(n):
            for j in range(i, n):
                set_i = id_sets[i]
                set_j = id_sets[j]
                union = set_i | set_j
                intersection = set_i & set_j

                jaccard_sim = len(intersection) / len(union) if union else 1.0
                dissim = 1 - jaccard_sim

                matrices[f][i, j] = dissim
                matrices[f][j, i] = dissim  # symmetric

    for subset, matrix in matrices.items():
        print("Mean dissimilarity for subset {}: {:.4f}".format(subset, np.mean(matrix)))
        print("###################################################")
    return matrices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["dissimilarity"],)
    parser.add_argument("--locations", type=str, nargs="?", default="all")
    args = parser.parse_args()


    if args.locations == 'all':
        locations_ = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]
    else:
        try:
            locations_ = [getattr(Locations, args.locations)]
        except AttributeError:
            raise ValueError(f"Unknown location: {args.locations}")

    if args.task == "dissimilarity":
        dissimilarity_across_data_splits(get_base_config(base_, locations_, CV=Keys.stratified, ))
    else:
        raise ValueError(f"Unknown task: {args.task}. Use 'channel_names' or 'subjects_with_seizures'.")