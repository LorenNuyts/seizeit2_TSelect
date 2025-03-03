import os
import polars as pl


def leave_one_person_out(root_dir: str, included_locations: list[str] = None):
    all_subjects = []
    for location in os.listdir(root_dir):
        if included_locations is not None and location not in included_locations:
            continue
        location_path = os.path.join(root_dir, location)
        if os.path.isdir(location_path):  # Ensure it's a folder
            for subject in os.listdir(location_path):
                all_subjects.append(subject)

    for subject in all_subjects:
        train = all_subjects.copy()
        train.remove(subject)
        yield train, [subject]

def leave_one_seizure_out(X: pl.DataFrame, y: pl.Series):
    end_previous_seizure = None
    previous_fold_split = 0
    in_seizure = False
    for i in range(len(y)):
        if not in_seizure and y[i] == 1:
            in_seizure = True
            if end_previous_seizure is None:
                continue
            start_next_seizure = i
            new_fold_split = (end_previous_seizure + start_next_seizure) // 2
            X_train = pl.concat([X.slice(0, previous_fold_split), X.slice(new_fold_split, len(X) - new_fold_split)])
            y_train = pl.concat([y.slice(0, previous_fold_split), y.slice(new_fold_split, len(y) - new_fold_split)])
            X_test = X.slice(previous_fold_split, new_fold_split - previous_fold_split)
            y_test = y.slice(previous_fold_split, new_fold_split - previous_fold_split)
            previous_fold_split = new_fold_split

            yield X_train, y_train, X_test, y_test

        elif in_seizure and y[i] == 0:
            in_seizure = False
            end_previous_seizure = i