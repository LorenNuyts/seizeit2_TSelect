"""Sensitivity of the selected channels to TSelect's correlation threshold.

The saved TSelect objects keep `rank_correlation` and `evaluation_metric_per_channel`, so the
redundancy-filter step can be replayed for any threshold without retraining anything. Only the
clustering changes; the irrelevant-filter output is whatever was stored in the run.

    python -m analysis.corr_threshold_sensitivity \\
        --results net/save_dir/results/<experiment>__all_results.pkl

Caveats:
  * This reports selection FREQUENCY, not rho_c. rho_c weights each combination by the score
    improvement of a model trained on it, and those models only exist for the threshold the run
    actually used.
  * The stored rank correlations come from whatever evaluation metric that run used. After
    changing the metric, re-run the selection before trusting this sweep.
"""
import argparse
import pickle
from collections import Counter
from operator import itemgetter

import numpy as np

from TSelect.tselect.tselect.rank_correlation.rank_correlation import cluster_correlations

parser = argparse.ArgumentParser()
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--results", help="path to a *__all_results.pkl file (selectors from a stored run)")
src.add_argument("--selectors", help="path written by analysis.dry_run_channel_selection "
                                     "--save-selector (selectors from a fresh run)")
parser.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
args = parser.parse_args()

if args.results:
    with open(args.results, 'rb') as fh:
        obj = pickle.load(fh)
    cfg = obj['config'] if isinstance(obj, dict) else obj.config
    channels = sorted(cfg.included_channels)
    selectors = getattr(cfg, 'channel_selector', None) or {}
    label = cfg.get_name()
    used = cfg.channel_selection_settings['corr_threshold']
    metric = cfg.channel_selection_settings['evaluation_metric'].__name__
else:
    with open(args.selectors, 'rb') as fh:
        payload = pickle.load(fh)
    channels = sorted(payload['included_channels'])
    selectors = payload['selectors']
    label = "dry run: %s" % args.selectors
    used = payload['corr_threshold']
    metric = payload['metric']

if not selectors:
    raise SystemExit("No channel_selector objects found in the given file.")

print("Source        :", label)
print("Threshold used:", used)
print("Metric used   :", metric)


def replay(selector, threshold):
    """Re-run clustering + best-per-cluster selection at the given threshold."""
    chosen = []
    for cluster in cluster_correlations(selector.rank_correlation, None, threshold=threshold):
        cluster = list(cluster)
        scores = itemgetter(*cluster)(selector.evaluation_metric_per_channel)
        scores = (scores,) if not isinstance(scores, tuple) else scores
        chosen.append(cluster[int(np.argmax(scores))])
    return [channels[i] for i in chosen]


header = "  ".join("%-5s" % t for t in args.thresholds)
per_fold, freq = {}, {t: Counter() for t in args.thresholds}
for fold in sorted(selectors):
    sel = selectors[fold]
    if not getattr(sel, 'rank_correlation', None):
        print("fold %d: no rank_correlation stored, skipped" % fold)
        continue
    per_fold[fold] = {t: replay(sel, t) for t in args.thresholds}
    for t in args.thresholds:
        freq[t].update(per_fold[fold][t])

print("\nNumber of channels selected per fold")
print("%5s | %s" % ("fold", header))
print("-" * (8 + 7 * len(args.thresholds)))
for fold in sorted(per_fold):
    print("%5d | %s" % (fold, "  ".join("%-5d" % len(per_fold[fold][t]) for t in args.thresholds)))
print("-" * (8 + 7 * len(args.thresholds)))
print("%5s | %s" % ("mean", "  ".join(
    "%-5.1f" % np.mean([len(per_fold[f][t]) for f in per_fold]) for t in args.thresholds)))

print("\nNumber of folds (out of %d) in which each channel is selected" % len(per_fold))
print("%-14s %s" % ("channel", header))
print("-" * (16 + 7 * len(args.thresholds)))
for ch in sorted(channels, key=lambda c: -sum(freq[t][c] for t in args.thresholds)):
    if sum(freq[t][ch] for t in args.thresholds) == 0:
        continue
    print("%-14s %s" % (ch, "  ".join("%-5d" % freq[t][ch] for t in args.thresholds)))
