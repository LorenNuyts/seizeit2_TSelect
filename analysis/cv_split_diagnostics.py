"""
Diagnostics for the stratified cross-validation splits.

By default the folds are read back from the configs of the trained runs, which store the subject
lists of every fold (see Config.save_config). Those are the folds the models were actually trained
on. Alternatively the folds can be regenerated with the splitter, which produces the same folds
because the splitter is deterministic given SEED.

The script writes the folds as JSON, a markdown report and the raw per-fold counts as CSV. It only
measures, it does not change the splitter and it does not interpret the results.

Usage:
    python -m analysis.cv_split_diagnostics
    python -m analysis.cv_split_diagnostics --source regenerate
    python -m analysis.cv_split_diagnostics --config-file <path to a .cfg file>
"""
import argparse
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from analysis.dataset import dataset_stats
from data import cross_validation as cross_validation_module
from data.cross_validation import get_CV_generator
from net.DL_config import get_base_config
from utility.constants import SEED, Keys, Locations, excluded_subjects, subjects_Fz_reference

base_ = os.path.dirname(os.path.realpath(__file__))

SUBSET_NAMES = {0: 'train', 1: 'validation', 2: 'test'}
SUBSET_KEYS = {0: 'train', 1: 'validation', 2: 'test'}  # keys used in the stored folds

# Totals the configuration must reproduce, as reported in the paper.
EXPECTED_TOTALS = {'subjects': 184, 'seizure_patients': 42, 'seizures': 370, 'hours': 6899}
HOURS_TOLERANCE = 1.0  # the expected number of hours is rounded to the nearest hour

# Number of random draws used for the unstratified baseline in section 4.
N_MONTE_CARLO = 1000


def get_diagnostics_config():
    """
    Builds the config of the cross-validation strategy of the published runs: all six hospital
    locations, a held-out fold, 10 folds, 80/10/10 proportions and no Fz reference.
    """
    locations = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                 Locations.leuven_adult, Locations.leuven_pediatric]
    config = get_base_config(base_, locations, CV=Keys.stratified, held_out_fold=True,
                             Fz_reference=False)
    assert config.n_folds == 10, "Expected 10 folds, got {}".format(config.n_folds)
    assert config.train_percentage == 0.8, \
        "Expected a training proportion of 0.8, got {}".format(config.train_percentage)
    assert config.validation_percentage == 0.1, \
        "Expected a validation proportion of 0.1, got {}".format(config.validation_percentage)
    return config


def check_splitter_is_not_in_testing_mode():
    """
    multi_objective_grouped_stratified_cross_validation falls back to a hard-coded three-subject
    debug split when the source tree does not live under a path containing 'dtai'. That branch does
    not produce the folds of the experiments, so refuse to regenerate anything in that case.
    """
    splitter_dir = os.path.dirname(os.path.realpath(cross_validation_module.__file__))
    if 'dtai' not in splitter_dir:
        raise RuntimeError(
            "The splitter runs in testing mode from this location ({}) and yields a hard-coded "
            "three-subject debug split instead of the real folds. Read the folds back from the "
            "trained configs instead (--source saved), or regenerate them on the cluster.".format(
                splitter_dir))


def find_saved_folds(config, config_file=None):
    """
    Reads the folds back from the configs of the trained runs.

    Collects every stored config that matches the cross-validation settings of the given config and
    holds the expected number of folds. All matching configs must agree on the folds and on the
    held-out subjects; otherwise the stored folds are ambiguous and the caller has to pick one with
    config_file.

    Returns the folds, the held-out subjects and the provenance of the folds.
    """
    if config_file is not None:
        candidate_paths = [config_file]
    else:
        candidate_paths = sorted(glob.glob(
            os.path.join(config.save_dir, 'models', '*', 'configs', '*.cfg')))

    matches = []
    for path in candidate_paths:
        try:
            with open(path, 'rb') as stored:
                stored_config = pickle.load(stored)
        except Exception:  # a config of an older version of the code
            continue
        folds = stored_config.get('folds') or {}
        if (stored_config.get('cross_validation') != config.cross_validation
                or stored_config.get('held_out_fold') != config.held_out_fold
                or stored_config.get('Fz_reference') != config.Fz_reference
                or set(stored_config.get('locations') or []) != set(config.locations)
                or len(folds) != config.n_folds):
            continue
        matches.append((path, stored_config, folds))

    if not matches:
        raise RuntimeError(
            "No stored config under {} holds {} folds for this cross-validation setting "
            "(cross_validation={}, held_out_fold={}, Fz_reference={}, {} locations). Train a model "
            "with this setting first, or regenerate the folds with --source regenerate.".format(
                os.path.join(config.save_dir, 'models'), config.n_folds, config.cross_validation,
                config.held_out_fold, config.Fz_reference, len(config.locations)))

    def signature(folds, held_out):
        return (tuple(sorted((fold_i, tuple(sorted(subsets['train'])),
                              tuple(sorted(subsets['validation'])), tuple(sorted(subsets['test'])))
                             for fold_i, subsets in folds.items())),
                tuple(sorted(held_out or [])))

    signatures = {signature(folds, stored_config.get('held_out_subjects'))
                  for _, stored_config, folds in matches}
    if len(signatures) != 1:
        raise RuntimeError(
            "The {} stored configs with this cross-validation setting do not agree on the folds "
            "({} different fold sets). Pass --config-file to choose one explicitly.".format(
                len(matches), len(signatures)))

    path, stored_config, folds = matches[0]
    splits = [tuple(list(folds[fold_i][SUBSET_KEYS[subset_i]]) for subset_i in sorted(SUBSET_NAMES))
              for fold_i in sorted(folds)]
    held_out_subjects = list(stored_config.get('held_out_subjects') or [])
    provenance = {'source': 'saved',
                  'config_file': path,
                  'n_matching_configs': len(matches),
                  'cross_validation': stored_config.get('cross_validation'),
                  'held_out_fold': stored_config.get('held_out_fold'),
                  'Fz_reference': stored_config.get('Fz_reference'),
                  'locations': list(stored_config.get('locations') or [])}
    return splits, held_out_subjects, provenance


def regenerate_folds(config):
    """ Regenerates the folds with the splitter. """
    check_splitter_is_not_in_testing_mode()
    CV_generator, held_out_subjects = get_CV_generator(config)
    splits = [tuple(list(subset) for subset in split) for split in CV_generator]
    provenance = {'source': 'regenerate',
                  'cross_validation': config.cross_validation,
                  'held_out_fold': config.held_out_fold,
                  'Fz_reference': config.Fz_reference,
                  'locations': list(config.locations),
                  'SEED': SEED}
    return splits, list(held_out_subjects or []), provenance


def get_subject_table(config, pool=None):
    """
    Builds the per-subject statistics table: the dataset statistics of the configured locations,
    without the Fz-reference subjects and without the excluded subjects. When a pool of subjects is
    given, the table is restricted to those subjects, which is what the folds are drawn from once
    the held-out subjects have been removed.
    """
    info_per_subject = dataset_stats(config.data_path, os.path.join(config.save_dir, "dataset_stats"),
                                     config.locations)
    if config.Fz_reference:
        info_per_subject = info_per_subject[info_per_subject['subject'].isin(subjects_Fz_reference)]
    else:
        info_per_subject = info_per_subject[~info_per_subject['subject'].isin(subjects_Fz_reference)]
    info_per_subject = info_per_subject[~info_per_subject['subject'].isin(excluded_subjects)]
    info_per_subject = info_per_subject.drop_duplicates(subset='subject').set_index('subject')
    if pool is not None:
        missing = set(pool) - set(info_per_subject.index)
        assert not missing, \
            "The folds contain subjects without dataset statistics: {}".format(sorted(missing))
        info_per_subject = info_per_subject.loc[sorted(pool)]
    return info_per_subject


def check_totals(subject_table):
    """
    Compares the totals of the subject table with the totals reported in the paper. Returns the
    comparison as a DataFrame and whether every total matches.
    """
    observed = {'subjects': subject_table.shape[0],
                'seizure_patients': int((subject_table['n_seizures'] > 0).sum()),
                'seizures': int(subject_table['n_seizures'].sum()),
                'hours': float(subject_table['hours_of_data'].sum())}
    rows = []
    for quantity, expected in EXPECTED_TOTALS.items():
        tolerance = HOURS_TOLERANCE if quantity == 'hours' else 0
        matches = abs(observed[quantity] - expected) <= tolerance
        rows.append({'quantity': quantity, 'expected': expected,
                     'observed': round(observed[quantity], 1) if quantity == 'hours' else observed[quantity],
                     'match': 'yes' if matches else 'NO'})
    comparison = pd.DataFrame(rows)
    return comparison, bool((comparison['match'] == 'yes').all())


def summarize_subset(subject_table, subject_ids):
    """ Counts of the given subjects: subjects, seizure patients, seizures and hours of data. """
    subject_ids = list(subject_ids)
    unknown = set(subject_ids) - set(subject_table.index)
    assert not unknown, \
        "The folds contain subjects that are not in the dataset statistics: {}".format(sorted(unknown))
    subset = subject_table.loc[subject_ids]
    return {'n_subjects': subset.shape[0],
            'n_seizure_patients': int((subset['n_seizures'] > 0).sum()),
            'n_seizures': int(subset['n_seizures'].sum()),
            'hours': round(float(subset['hours_of_data'].sum()), 1)}


def realised_subset_sizes(splits, subject_table):
    """ Section 1: the realised size of every subset in every fold. """
    rows = []
    for fold_i, split in enumerate(splits):
        for subset_i, subset_name in SUBSET_NAMES.items():
            row = {'fold': fold_i, 'subset': subset_name}
            row.update(summarize_subset(subject_table, split[subset_i]))
            rows.append(row)
    per_fold = pd.DataFrame(rows)

    metrics = ['n_subjects', 'n_seizure_patients', 'n_seizures', 'hours']
    # 'std' is the sample standard deviation over the folds (pandas default, ddof=1).
    summary = per_fold.groupby('subset')[metrics].agg(['mean', 'std', 'min', 'max']).round(1)
    summary = summary.reindex([SUBSET_NAMES[i] for i in sorted(SUBSET_NAMES)])
    summary.columns = ["{} ({})".format(metric, statistic) for metric, statistic in summary.columns]
    return per_fold, summary


def occupancy(splits, subject_table, subset_i):
    """
    Section 2: how often each subject occurs in the given subset across the folds, and what is never
    covered.
    """
    counts = Counter({subject: 0 for subject in subject_table.index})
    for split in splits:
        counts.update(split[subset_i])

    counts_per_subject = pd.Series(counts, name='n_appearances').sort_index()
    seizure_patients = subject_table.index[subject_table['n_seizures'] > 0]

    def distribution(subjects):
        appearances = counts_per_subject.loc[list(subjects)]
        return {'0 times': int((appearances == 0).sum()),
                '1 time': int((appearances == 1).sum()),
                '2 times': int((appearances == 2).sum()),
                '3+ times': int((appearances >= 3).sum()),
                'max': int(appearances.max()),
                'total': len(subjects)}

    table = pd.DataFrame([{'group': 'all subjects', **distribution(subject_table.index)},
                          {'group': 'seizure patients', **distribution(seizure_patients)}])

    never_covered = counts_per_subject.index[counts_per_subject == 0]
    never = summarize_subset(subject_table, never_covered)
    return table, counts_per_subject, never


def per_hospital_seizures(splits, subject_table):
    """ Section 3: the number of seizures per hospital in the validation and test block of each fold. """
    hospitals = sorted(subject_table['hospital'].unique())
    rows = []
    for fold_i, split in enumerate(splits):
        for subset_i in (1, 2):
            subset = subject_table.loc[list(split[subset_i])]
            seizures = subset.groupby('hospital')['n_seizures'].sum()
            row = {'fold': fold_i, 'subset': SUBSET_NAMES[subset_i]}
            row.update({hospital: int(seizures.get(hospital, 0)) for hospital in hospitals})
            rows.append(row)
    per_fold = pd.DataFrame(rows)

    zero_rows = []
    for subset_i in (1, 2):
        subset_name = SUBSET_NAMES[subset_i]
        folds = per_fold[per_fold['subset'] == subset_name]
        row = {'subset': subset_name}
        row.update({hospital: int((folds[hospital] == 0).sum()) for hospital in hospitals})
        zero_rows.append(row)
    n_folds_without_seizures = pd.DataFrame(zero_rows)
    return per_fold, n_folds_without_seizures


def mean_pairwise_jaccard(id_sets):
    """ Mean Jaccard overlap over all unordered pairs of the given collections of subject IDs. """
    overlaps = []
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            union = id_sets[i] | id_sets[j]
            overlaps.append(len(id_sets[i] & id_sets[j]) / len(union) if union else 1.0)
    return float(np.mean(overlaps))


def max_attainable_dissimilarity(sizes, pool_size):
    """
    The largest mean pairwise Jaccard dissimilarity that subsets of the given realised sizes can
    reach within a pool of the given size. Two subsets of sizes a and b out of a pool of N share at
    least max(0, a + b - N) subjects, so their Jaccard overlap is at least
    max(0, a + b - N) / min(a + b, N).
    """
    minima = []
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            smallest_intersection = max(0, sizes[i] + sizes[j] - pool_size)
            largest_union = min(sizes[i] + sizes[j], pool_size)
            minima.append(smallest_intersection / largest_union if largest_union else 0.0)
    return float(1 - np.mean(minima))


def unstratified_baseline(splits, subject_table, subset_i, n_draws=N_MONTE_CARLO, seed=SEED):
    """
    Section 4: the mean pairwise Jaccard overlap that subsets of the realised sizes would reach if
    they were drawn uniformly at random from the subject pool, without stratification.
    """
    rng = np.random.default_rng(seed)
    subjects = np.array(subject_table.index)
    sizes = [len(split[subset_i]) for split in splits]

    means = []
    for _ in range(n_draws):
        id_sets = [set(rng.choice(subjects, size=size, replace=False)) for size in sizes]
        means.append(mean_pairwise_jaccard(id_sets))
    return float(np.mean(means)), float(np.std(means))


def within_fold_leakage(splits, held_out_subjects):
    """
    Section 5: the number of subjects shared by two subsets of the same fold, and between a fold and
    the held-out subjects. Should be zero everywhere.
    """
    held_out = set(held_out_subjects)
    rows = []
    for fold_i, split in enumerate(splits):
        sets = {name: set(split[i]) for i, name in SUBSET_NAMES.items()}
        rows.append({'fold': fold_i,
                     'train & validation': len(sets['train'] & sets['validation']),
                     'train & test': len(sets['train'] & sets['test']),
                     'validation & test': len(sets['validation'] & sets['test']),
                     'fold & held-out': len((sets['train'] | sets['validation'] | sets['test']) & held_out)})
    return pd.DataFrame(rows)


def check_pool_per_fold(splits):
    """
    Every fold must cover the same pool of subjects. Returns that pool.
    """
    pools = [set(split[0]) | set(split[1]) | set(split[2]) for split in splits]
    for fold_i, pool in enumerate(pools):
        assert pool == pools[0], \
            "Fold {} covers {} subjects while fold 0 covers {}".format(fold_i, len(pool), len(pools[0]))
    return pools[0]


def export_folds_json(path, splits, held_out_subjects, provenance, totals):
    """ Writes the folds, the held-out subjects and their provenance to a self-describing JSON file. """
    document = {
        'provenance': provenance,
        'totals': {row['quantity']: row['observed'] for _, row in totals.iterrows()},
        'expected_totals': EXPECTED_TOTALS,
        'held_out_subjects': sorted(held_out_subjects),
        'n_folds': len(splits),
        'folds': [{'fold': fold_i,
                   'train': sorted(split[0]),
                   'validation': sorted(split[1]),
                   'test': sorted(split[2])}
                  for fold_i, split in enumerate(splits)],
    }
    with open(path, 'w') as output:
        json.dump(document, output, indent=2)
    return path


def to_markdown(df, index=False):
    return df.to_markdown(index=index)


def run(config, output_dir, source='saved', config_file=None, n_draws=N_MONTE_CARLO,
        ignore_totals_mismatch=False):
    os.makedirs(output_dir, exist_ok=True)

    if source == 'saved':
        splits, held_out_subjects, provenance = find_saved_folds(config, config_file=config_file)
    elif source == 'regenerate':
        splits, held_out_subjects, provenance = regenerate_folds(config)
    else:
        raise ValueError("Unknown source: {}".format(source))

    assert len(splits) == config.n_folds, \
        "Expected {} folds, got {}".format(config.n_folds, len(splits))
    assert all(len(split) == 3 for split in splits), "Expected three subsets per fold"
    pool = check_pool_per_fold(splits)

    subject_table = get_subject_table(config, pool=pool)
    totals, totals_match = check_totals(subject_table)
    print(to_markdown(totals))
    if not totals_match and not ignore_totals_mismatch:
        raise RuntimeError(
            "The totals of the subjects covered by these folds do not match the totals reported in "
            "the paper (see the table above), so these are not the folds of the published runs. "
            "Pass --ignore-totals-mismatch to measure anyway.")

    json_path = export_folds_json(os.path.join(output_dir, 'cv_folds.json'), splits,
                                  held_out_subjects, provenance, totals)

    sizes_per_fold, sizes_summary = realised_subset_sizes(splits, subject_table)
    test_occupancy, test_counts, test_never = occupancy(splits, subject_table, 2)
    val_occupancy, val_counts, val_never = occupancy(splits, subject_table, 1)
    seizures_per_hospital, folds_without_seizures = per_hospital_seizures(splits, subject_table)
    leakage = within_fold_leakage(splits, held_out_subjects)

    n_splits = len(splits)
    observed_overlap = {name: mean_pairwise_jaccard([set(split[i]) for split in splits])
                        for i, name in SUBSET_NAMES.items()}
    ceilings = {name: max_attainable_dissimilarity([len(split[i]) for split in splits], len(pool))
                for i, name in SUBSET_NAMES.items()}

    baseline_mean, baseline_std = unstratified_baseline(splits, subject_table, 2, n_draws=n_draws)
    overlap_table = pd.DataFrame([
        {'quantity': 'observed mean pairwise Jaccard overlap of the test blocks',
         'value': round(observed_overlap['test'], 4)},
        {'quantity': 'observed mean pairwise Jaccard dissimilarity of the test blocks',
         'value': round(1 - observed_overlap['test'], 4)},
        {'quantity': 'expected overlap for unstratified draws of the same realised sizes '
                     '(mean over {} draws)'.format(n_draws),
         'value': round(baseline_mean, 4)},
        {'quantity': 'standard deviation over the {} draws'.format(n_draws),
         'value': round(baseline_std, 4)},
    ])
    dissimilarity_table = pd.DataFrame([
        {'subset': name,
         'observed overlap': round(observed_overlap[name], 4),
         'observed dissimilarity': round(1 - observed_overlap[name], 4),
         'max attainable dissimilarity at the realised sizes': round(ceilings[name], 4)}
        for name in [SUBSET_NAMES[i] for i in sorted(SUBSET_NAMES)]])

    sizes_per_fold.to_csv(os.path.join(output_dir, 'per_fold_subset_sizes.csv'), index=False)
    seizures_per_hospital.to_csv(os.path.join(output_dir, 'per_fold_seizures_per_hospital.csv'),
                                 index=False)
    appearances = pd.DataFrame({'test_blocks': test_counts, 'validation_blocks': val_counts})
    appearances = subject_table[['hospital', 'n_seizures', 'hours_of_data']].join(appearances)
    appearances.to_csv(os.path.join(output_dir, 'per_subject_appearances.csv'))

    report_path = os.path.join(output_dir, 'cv_split_diagnostics.md')
    with open(report_path, 'w') as report:
        report.write("# Cross-validation split diagnostics\n\n")

        report.write("## Configuration\n\n")
        configuration = pd.DataFrame([
            {'setting': 'folds obtained by', 'value': provenance['source']},
            {'setting': 'source config file', 'value': provenance.get('config_file', '-')},
            {'setting': 'stored configs agreeing on these folds',
             'value': provenance.get('n_matching_configs', '-')},
            {'setting': 'cross-validation', 'value': config.cross_validation},
            {'setting': 'locations', 'value': ", ".join(config.locations)},
            {'setting': 'held_out_fold', 'value': config.held_out_fold},
            {'setting': 'held-out subjects', 'value': len(held_out_subjects)},
            {'setting': 'subjects the folds are drawn from', 'value': len(pool)},
            {'setting': 'n_folds', 'value': config.n_folds},
            {'setting': 'proportions (train/validation/test)',
             'value': "{}/{}/{}".format(config.train_percentage, config.validation_percentage,
                                        round(1 - config.train_percentage - config.validation_percentage, 10))},
            {'setting': 'SEED', 'value': SEED},
            {'setting': 'Fz_reference', 'value': config.Fz_reference},
            {'setting': 'Fz-reference subjects removed', 'value': len(subjects_Fz_reference)},
            {'setting': 'excluded subjects', 'value': len(excluded_subjects)},
            {'setting': 'data_path', 'value': config.data_path},
            {'setting': 'Monte Carlo draws (section 4)', 'value': n_draws},
        ])
        report.write(to_markdown(configuration) + "\n\n")
        report.write("Totals of the {} subjects the folds are drawn from, against the totals "
                     "reported in the paper:\n\n".format(len(pool)))
        report.write(to_markdown(totals) + "\n\n")
        report.write("The folds are written to `{}`.\n\n".format(os.path.basename(json_path)))

        report.write("## 1. Realised subset sizes\n\n")
        report.write("Per fold:\n\n")
        report.write(to_markdown(sizes_per_fold) + "\n\n")
        report.write("Across folds (std is the sample standard deviation, ddof=1):\n\n")
        report.write(to_markdown(sizes_summary, index=True) + "\n\n")

        report.write("## 2. Test-block coverage\n\n")
        report.write("Number of test blocks a subject appears in, over the {} folds:\n\n".format(n_splits))
        report.write(to_markdown(test_occupancy) + "\n\n")
        report.write("Subjects that never appear in a test block: {} subjects, "
                     "{} seizure patients, {} seizures, {} hours.\n\n".format(
                         test_never['n_subjects'], test_never['n_seizure_patients'],
                         test_never['n_seizures'], test_never['hours']))
        report.write("The same counts for the validation blocks:\n\n")
        report.write(to_markdown(val_occupancy) + "\n\n")
        report.write("Subjects that never appear in a validation block: {} subjects, "
                     "{} seizure patients, {} seizures, {} hours.\n\n".format(
                         val_never['n_subjects'], val_never['n_seizure_patients'],
                         val_never['n_seizures'], val_never['hours']))

        report.write("## 3. Per-hospital representation\n\n")
        report.write("Number of seizures per hospital in the validation and test block of each fold:\n\n")
        report.write(to_markdown(seizures_per_hospital) + "\n\n")
        report.write("Number of folds (out of {}) whose block contains zero seizures "
                     "from a hospital:\n\n".format(n_splits))
        report.write(to_markdown(folds_without_seizures) + "\n\n")

        report.write("## 4. Observed vs. independent-draw overlap\n\n")
        report.write("Mean over the {} unordered pairs of folds, on the realised test-block "
                     "sizes:\n\n".format(n_splits * (n_splits - 1) // 2))
        report.write(to_markdown(overlap_table) + "\n\n")
        report.write("Per subset, against the largest dissimilarity that subsets of the realised "
                     "sizes can reach within a pool of {} subjects:\n\n".format(len(pool)))
        report.write(to_markdown(dissimilarity_table) + "\n\n")

        report.write("## 5. Within-fold overlap between subsets\n\n")
        report.write("Number of subjects shared by two subsets of the same fold, and between a fold "
                     "and the held-out subjects:\n\n")
        report.write(to_markdown(leakage) + "\n\n")

    print("\nFolds written to", json_path)
    print("Report written to", report_path)
    return report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnostics for the stratified cross-validation splits.")
    parser.add_argument("--source", type=str, default="saved", choices=["saved", "regenerate"],
                        help="Read the folds back from the configs of the trained runs ('saved', "
                             "the default) or regenerate them with the splitter ('regenerate').")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Read the folds from this .cfg file instead of searching the models "
                             "directory. Only used with --source saved.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for the folds, the report and the CSV files. Defaults to "
                             "<save_dir>/cv_split_diagnostics.")
    parser.add_argument("--n-draws", type=int, default=N_MONTE_CARLO,
                        help="Number of random draws for the unstratified baseline of section 4.")
    parser.add_argument("--ignore-totals-mismatch", action="store_true",
                        help="Measure even when the totals do not match the totals reported in the "
                             "paper.")
    args = parser.parse_args()

    config_ = get_diagnostics_config()
    output_dir_ = args.output_dir or os.path.join(config_.save_dir, 'cv_split_diagnostics')
    run(config_, output_dir_, source=args.source, config_file=args.config_file,
        n_draws=args.n_draws, ignore_totals_mismatch=args.ignore_totals_mismatch)
