import argparse
import os

import numpy as np

from data.cross_validation import get_CV_generator
from net.DL_config import get_base_config
from utility.constants import Keys, Locations

base_ = os.path.dirname(os.path.realpath(__file__))

def dissimilarity_across_data_splits(config, verbose: bool = True):
    """
    Computes how dissimilar the subsets (train, validation, test) are across the folds of the
    cross-validation strategy of the given config.

    Args:
        config (cls): a config object determining the cross-validation strategy
        verbose (bool): whether to print the mean dissimilarity per subset

    Returns: dict mapping the subset index (0 = train, 1 = validation, 2 = test) to a symmetric
        2D numpy array (n_splits x n_splits) with the pairwise Jaccard dissimilarity between the
        folds. The diagonal is zero.
    """
    CV_generator, _ = get_CV_generator(config)
    splits = list(CV_generator)
    n_splits = len(splits)
    n_subsets = len(splits[0])
    dissimilarities = {s: np.zeros((n_splits, n_splits)) for s in range(n_subsets)}

    for s in range(n_subsets):
        id_sets = [set(split[s]) for split in splits]

        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                set_i = id_sets[i]
                set_j = id_sets[j]
                union = set_i | set_j
                intersection = set_i & set_j

                jaccard_sim = len(intersection) / len(union) if union else 1.0
                dissim = 1 - jaccard_sim

                dissimilarities[s][i, j] = dissim
                dissimilarities[s][j, i] = dissim

    if verbose:
        for subset, dissim in dissimilarities.items():
            if n_splits < 2:
                print("Only one split, so no dissimilarity can be computed for subset {}".format(subset))
            else:
                # Only the pairs above the diagonal are distinct pairs
                mean_dissim = dissim[np.triu_indices(n_splits, k=1)].mean()
                print("Mean dissimilarity for subset {}: {:.4f}".format(subset, mean_dissim))
            print("###################################################")

    return dissimilarities


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