import math
import os
import random
from typing import Optional

import numpy as np

from utility.constants import SEED, subjects_with_seizures, excluded_subjects


# subjects_with_seizures = ['SUBJ-7-331', 'SUBJ-7-379', 'SUBJ-7-376', 'SUBJ-7-438', 'SUBJ-7-441', 'SUBJ-7-449']

def leave_one_person_out(root_dir: str, included_locations: list[str] = None, validation_set: Optional[float] = None,
                         seed: int = SEED):
    testing = 'dtai' not in root_dir
    print("Testing setting:", testing)
    if testing:
        nb_subjects = 3 # ONLY FOR TESTING
        # included_subjects = ['SUBJ-7-331', 'SUBJ-7-379', 'SUBJ-7-376'] # Coimbra subjects
        included_subjects = ['SUBJ-1a-159', 'SUBJ-1a-358', 'SUBJ-1a-153'] # Leuven Adult subjects
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

