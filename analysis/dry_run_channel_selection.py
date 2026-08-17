"""Run TSelect for a single fold and print the channel scores, without training ChronoNet.

Reproduces the channel-selection step of main_func.train (lines 78-186) for one fold, then stops.
Reuses the cached segment pickles and TFRecords of the named experiment, so it is fast.

WRITES NOTHING: the config, results and model directories of the experiment are never touched, so
this is safe to point at the run behind the paper.

Example -- fold 3 of the run behind Table 4 (= "Fold 4" in the paper, selected T7 + F7):

    python -m analysis.dry_run_channel_selection --fold 3 \
        --evaluation_metric score --irr_th 0 --auc 0.3 --corr 0.5 \
        --held_out_fold --CV stratified

Leave --suffix empty: `Config.get_name` already appends `_v{CURRENT_VERSION}`, so passing
`--suffix v2` produces `..._v2_v2`, which does not match the stored experiment. The fold split is
regenerated identically either way (it depends only on SEED and the subject list), but the stored
config, the cached segments and the previous selection are then not found, so the run recomputes
the segments and cannot print the PREVIOUS/CHANGED comparison.
"""
import argparse
import os

from utility.constants import evaluation_metrics, parse_location, Locations, Keys

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--model", type=str, default="ChronoNet")
parser.add_argument("--evaluation_metric", type=str, default="score", choices=list(evaluation_metrics.keys()))
parser.add_argument("--nodes", type=str, default="all")
parser.add_argument("--irr_th", type=float, default=0.0)
parser.add_argument("--auc", type=float, default=0.3)
parser.add_argument("--corr", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--suffix", type=str, default="",
                    help="extra suffix for the experiment name. Leave empty: get_name() already "
                         "appends _v{CURRENT_VERSION}, so 'v2' here yields '..._v2_v2'.")
parser.add_argument("--CV", type=str, default=Keys.stratified,
                    choices=["leave_one_person_out", "stratified", "leave_one_hospital_out"])
parser.add_argument("--held_out_fold", action="store_true")
parser.add_argument("--Fz_reference", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--save-selector", type=str, default=None,
                    help="path to accumulate the fitted selectors into, for later analysis "
                         "(e.g. analysis.corr_threshold_sensitivity --selectors). Running several "
                         "folds with the same path merges them. This is a scratch file; it never "
                         "touches the experiment's own config or results.")
parser.add_argument("--locations", nargs="+", type=parse_location,
                    default=[parse_location(l) for l in Locations.all_keys()])
parser.add_argument("--debug", dest='debug', action='store_true', default=False,
                    help="Run on the handful of subjects of a debug run instead of the whole "
                         "dataset (see utility/debug_settings.py). Opt-in, unlike the entry points "
                         "that train: this script looks the experiment up by name, so a debug run "
                         "only finds the stored config of a run that was itself a debug run.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pickle

import numpy as np

from data.cross_validation import get_CV_generator
from net.DL_config import get_channel_selection_config
from net.generator_ds import build_tfrecord_dataset
from net.key_generator import generate_data_keys_subsample, generate_data_keys_sequential_window
from utility import get_recs_list
from utility.constants import SEED
from utility.debug_settings import get_debug_settings
from utility.paths import (get_path_config, get_path_results, get_paths_segments_train,
                           get_paths_segments_val)

from TSelect.tselect.tselect.utils import init_metadata
from TSelect.tselect.tselect.channel_selectors.tselect import TSelect

base_ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

config = get_channel_selection_config(
    base_, sorted(list(dict.fromkeys(args.locations))), model=args.model,
    evaluation_metric=evaluation_metrics[args.evaluation_metric],
    irrelevant_selector_threshold=args.irr_th, irrelevant_selector_percentage=args.auc,
    corr_threshold=args.corr, CV=args.CV, suffix=args.suffix, included_channels=args.nodes,
    batch_size=args.batch_size, held_out_fold=args.held_out_fold, Fz_reference=args.Fz_reference,
    debug=get_debug_settings(True) if args.debug else None)

name = config.get_name()
print("Experiment:", name)
print("Metric    :", config.channel_selection_settings['evaluation_metric'].__name__)
print("irr_th    :", args.irr_th, " top-p:", args.auc, " corr:", args.corr)

# Load the stored config so the folds match the original run exactly. Read only.
config_path = get_path_config(config, name)
if os.path.exists(os.path.join(config_path, name + '.cfg')):
    config.load_config(config_path, name)
    print("Loaded stored config (folds and paths reused).")
else:
    print("No stored config found; folds are regenerated from SEED =", SEED)

previous = None
results_path = get_path_results(config, name)
if os.path.exists(results_path):
    with open(results_path, 'rb') as fh:
        stored = pickle.load(fh)
    stored_cfg = stored['config'] if isinstance(stored, dict) else stored.config
    previous = getattr(stored_cfg, 'selected_channels', {}) or {}

# ---------------------------------------------------------------- build the fold's generators
CV_generator, held_out_subjects = get_CV_generator(config)
config.held_out_subjects = held_out_subjects

target = None
for fold_i, split in enumerate(CV_generator):
    if fold_i == args.fold:
        target = split
        break
if target is None:
    raise SystemExit(f"Fold {args.fold} not produced by the CV generator.")
train_subjects, validation_subjects, test_subjects = target

print(f"\nFold {args.fold}: {len(train_subjects)} train / {len(validation_subjects)} val / "
      f"{len(test_subjects)} test subjects")

train_recs = get_recs_list(config.data_path, config.locations, train_subjects)
val_recs = get_recs_list(config.data_path, config.locations, validation_subjects)


def load_or_build(path, build):
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            print("  reusing cached segments:", os.path.basename(path))
            return pickle.load(fh)
    print("  generating segments (not cached)...")
    return build()


train_segments = load_or_build(get_paths_segments_train(config, name, args.fold),
                               lambda: generate_data_keys_subsample(config, train_recs))
val_segments = load_or_build(get_paths_segments_val(config, name, args.fold),
                             lambda: generate_data_keys_sequential_window(config, val_recs,
                                                                          config.val_batch_size))
print(f"  train segments: {len(train_segments)}   val segments: {len(val_segments)}")

gen_train, _ = build_tfrecord_dataset(config, train_recs, train_segments,
                                      batch_size=config.batch_size, shuffle=True)
gen_val, _ = build_tfrecord_dataset(config, val_recs, val_segments,
                                    batch_size=config.val_batch_size, shuffle=False)

# ---------------------------------------------------------------- run the selector only
selector = TSelect(random_state=SEED,
                   evaluation_metric=config.channel_selection_settings['evaluation_metric'],
                   irrelevant_selector_percentage=config.channel_selection_settings['irrelevant_selector_percentage'],
                   filtering_threshold_corr=config.channel_selection_settings['corr_threshold'],
                   irrelevant_selector_threshold=config.channel_selection_settings['irrelevant_selector_threshold'])
selector.fit_generator(gen_train, gen_val, metadata=init_metadata())

# ---------------------------------------------------------------- report
chs = sorted(config.included_channels)
kept = selector.evaluation_metric_per_channel
removed = dict(selector.removed_series_too_low_metric)
all_scores = {**removed, **kept}
selected = [chs[i] for i in selector.selected_channels]

print("\n" + "=" * 72)
print("PER-CHANNEL SCORES  (higher is better; irr_th = %s)" % args.irr_th)
print("=" * 72)
for ch_i in sorted(all_scores, key=lambda k: -all_scores[k]):
    flag = "kept   " if ch_i in kept else "REMOVED"
    print("  %-14s %10.5f   %s" % (chs[ch_i], all_scores[ch_i], flag))

vals = np.array([all_scores[i] for i in sorted(all_scores)])
uniq, cnt = np.unique(np.round(vals, 6), return_counts=True)
print("\ndistinct scores : %d / %d channels" % (len(uniq), len(vals)))
print("largest tie     : %d channels sharing %.5f" % (cnt.max(), uniq[cnt.argmax()]))
print("negative scores : %d  (these are the ones irr_th=0 discards)" % int((vals < 0).sum()))
if len(kept) == len(all_scores) and (vals < 0).any():
    print("  !! every channel was kept despite negative scores -- TSelect fell back to "
          "'no series passed the threshold' (tselect.py:526)")

print("\nclusters : %s" % [[chs[i] for i in cl] for cl in (selector.clusters or [])])
print("SELECTED : %s" % selected)
if previous and args.fold in previous:
    print("PREVIOUS : %s   <- stored run (old metric)" % previous[args.fold])
    print("CHANGED  : %s" % ("yes" if set(previous[args.fold]) != set(selected) else "no"))

if args.save_selector:
    payload = {'included_channels': chs, 'corr_threshold': config.channel_selection_settings['corr_threshold'],
               'metric': config.channel_selection_settings['evaluation_metric'].__name__, 'selectors': {}}
    if os.path.exists(args.save_selector):
        with open(args.save_selector, 'rb') as fh:
            payload = pickle.load(fh)
    payload['selectors'][args.fold] = selector
    os.makedirs(os.path.dirname(os.path.abspath(args.save_selector)), exist_ok=True)
    with open(args.save_selector, 'wb') as fh:
        pickle.dump(payload, fh, pickle.HIGHEST_PROTOCOL)
    print("\nSelector for fold %d saved to %s (folds now present: %s)"
          % (args.fold, args.save_selector, sorted(payload['selectors'])))

print("\n(the experiment's own config, results and models are untouched)")
