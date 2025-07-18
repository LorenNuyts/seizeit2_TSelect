import argparse
import os

import numpy as np

from data.cross_validation import get_CV_generator
from net.DL_config import get_base_config
from utility.constants import Keys, Locations

base_ = os.path.dirname(os.path.realpath(__file__))

def dissimilarity_across_data_splits(config):
    """
    splits: List of split dicts, where each dict has 'train', 'val', 'test' DataFrames with 'patient_id'
    Returns: dict of dissimilarity matrices (one per subset), each as a 2D numpy array (NxN)
    """
    splits = [x for x in get_CV_generator(config)]
    n = len(splits)
    n_folds = len(splits[0])
    dissimilarities = {i: [] for i in range(n_folds)}

    for f in range(n_folds):
        id_sets = [set(split[f]) for split in splits]

        for i in range(n):
            for j in range(i + 1, n):
                set_i = id_sets[i]
                set_j = id_sets[j]
                union = set_i | set_j
                intersection = set_i & set_j

                jaccard_sim = len(intersection) / len(union) if union else 1.0
                dissim = 1 - jaccard_sim

                dissimilarities[f].append(dissim)

    for subset, dissim in dissimilarities.items():
        print("Mean dissimilarity for subset {}: {:.4f}".format(subset, np.mean(dissim)))
        print("###################################################")


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