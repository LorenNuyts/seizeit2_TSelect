"""Measure the effect of different RMSA combination rules on the reported metrics.

Compares, per recording, the fraction of epochs the artefact filter keeps and the resulting
sensitivity / false alarm rate / score, for several ways of combining the per-channel masks:

    none        no RMSA filtering at all (reference)
    current     what the code does today: `rmsa = rmsa and rmsa_ch` keeps only the LAST channel
                in the preprocessed file (alphabetically last, i.e. T8 for the 21-channel files)
    all21       element-wise AND over every baseline channel
    wearables   element-wise AND over the wearable channels only
    selected    element-wise AND over the channels this fold's model actually uses

Run it on the machine that has the data, once per experiment:

    python -m analysis.rmsa_impact \\
        --results net/save_dir/results/<experiment>__all_results.pkl

Compare the channel-selection run against the F union W baseline run: if `all21` and `wearables`
diverge between the two, the artefact filter is acting as a confound in the Table 4 comparison and
the reference set should be held fixed across both arms.
"""
import argparse
import glob
import os
import pickle
import warnings

import h5py
import numpy as np

from data.data import Data
from utility.constants import Nodes
from utility.metrics import get_metrics_scoring
from utility.paths import get_path_predictions_folder

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True, help="path to a *__all_results.pkl file")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--max-recordings", type=int, default=None,
                    help="stop after this many recordings (per fold) for a quick look")
parser.add_argument("--rms-low", type=float, default=13.0)
parser.add_argument("--rms-high", type=float, default=150.0)
parser.add_argument("--rms-stats", action="store_true",
                    help="skip the metric comparison; instead report the per-channel RMS "
                         "distribution, to check whether the 13-150 uV band fits this data")
args = parser.parse_args()

with open(args.results, 'rb') as fh:
    obj = pickle.load(fh)
config = obj['config'] if isinstance(obj, dict) else obj.config
name = config.get_name()
print("Experiment:", name)
print("Threshold :", args.threshold, "  RMS band: %g-%g uV" % (args.rms_low, args.rms_high))

PRED_FS = 1  # predictions are one per second


def channel_rms(x, fs):
    """RMS over 2 s windows at a 1 s stride -- vectorised equivalent of the loop in main_func."""
    w, s = 2 * fs, 1 * fs
    x = np.asarray(x, dtype=np.float64)
    if len(x) < w:
        return np.zeros(0)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.arange(0, len(x) - w + 1, s)
    return np.sqrt((csum[starts + w] - csum[starts]) / w)


def masks_for(rec_data, fs, lo, hi):
    return {ch: ((r > lo) & (r < hi))
            for ch, r in ((c, channel_rms(rec_data.data[i], fs))
                          for i, c in enumerate(rec_data.channels))}


def combine(per_ch, names, n):
    """Element-wise AND over `names`, truncated/padded to n epochs."""
    present = [per_ch[c] for c in names if c in per_ch]
    if not present:
        return None
    m = np.logical_and.reduce([p[:n] for p in present]) if len(present) > 1 else present[0][:n]
    if len(m) < n:                      # mask shorter than the prediction array
        m = np.concatenate([m, np.ones(n - len(m), dtype=bool)])
    return m


RULES = ["none", "current", "all21", "wearables", "selected"]
acc = {r: {"kept": [], "sens": [], "fa": [], "score": [], "dead": 0} for r in RULES}
rms_pool = {}          # channel -> list of subsampled RMS values, for --rms-stats
n_done = 0

for fold in sorted(config.folds):
    folder = get_path_predictions_folder(config, name, fold)
    files = sorted(glob.glob(os.path.join(folder, "*__preds.h5")))
    if not files:
        continue
    selected = (config.selected_channels[fold] if getattr(config, 'channel_selection', False)
                else config.included_channels)
    if args.max_recordings:
        files = files[:args.max_recordings]
    print("\nfold %d: %d recordings   model channels: %s" % (fold, len(files), selected))

    for path in files:
        base = os.path.basename(path)
        rec = base.split('__')[:3]
        with h5py.File(path, 'r') as f:
            y_pred = np.array(f['y_pred'], dtype=float)
            y_true = np.array(f['y_true'], dtype=float)
        if len(y_pred) != len(y_true):
            warnings.warn("%s: len(y_pred)=%d != len(y_true)=%d, skipping"
                          % (base, len(y_pred), len(y_true)))
            continue

        rec_data = Data.loadData(config.data_path, rec, included_channels=config.included_channels)
        rec_data.apply_preprocess(config.fs, data_path=config.data_path,
                                  store_preprocessed=True, recording=rec)
        if args.rms_stats:
            for i, ch in enumerate(rec_data.channels):
                r = channel_rms(rec_data.data[i], config.fs)
                rms_pool.setdefault(ch, []).append(r[::10])   # subsample to bound memory
            n_done += 1
            del rec_data
            continue

        per_ch = masks_for(rec_data, config.fs, args.rms_low, args.rms_high)
        n = len(y_pred)

        variants = {
            "none":      np.ones(n, dtype=bool),
            "current":   combine(per_ch, [rec_data.channels[-1]], n),
            "all21":     combine(per_ch, sorted(Nodes.basic_eeg_nodes + Nodes.included_wearables), n),
            "wearables": combine(per_ch, Nodes.included_wearables, n),
            "selected":  combine(per_ch, list(selected), n),
        }

        for rule, mask in variants.items():
            if mask is None:
                continue
            yp = np.where(mask, y_pred, 0.0)
            sens, _, _, _, _, _, _, fa_epoch, _ = get_metrics_scoring(yp, y_true, PRED_FS, args.threshold)
            acc[rule]["kept"].append(float(mask.mean()))
            acc[rule]["sens"].append(sens)
            acc[rule]["fa"].append(fa_epoch)
            acc[rule]["score"].append(sens * 100 - 0.4 * fa_epoch)
            if mask.mean() == 0:
                acc[rule]["dead"] += 1
        n_done += 1
        del rec_data, per_ch

if args.rms_stats:
    print("\n" + "=" * 88)
    print("Per-channel RMS distribution over %d recordings (2 s windows, uV)" % n_done)
    print("band under test: %g - %g uV" % (args.rms_low, args.rms_high))
    print("=" * 88)
    print("%-14s %8s %8s %8s %8s %8s | %8s %8s %8s" %
          ("channel", "p5", "p25", "median", "p75", "p95", "in band%", "below%", "above%"))
    for ch in sorted(rms_pool):
        v = np.concatenate(rms_pool[ch])
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        q = np.percentile(v, [5, 25, 50, 75, 95])
        below = 100 * np.mean(v <= args.rms_low)
        above = 100 * np.mean(v >= args.rms_high)
        print("%-14s %8.1f %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f" %
              (ch, q[0], q[1], q[2], q[3], q[4], 100 - below - above, below, above))
    print("\nIf a channel's median sits below %g uV, the lower bound is rejecting normal signal "
          "for that\nchannel rather than flat/disconnected segments, and the band needs "
          "recalibrating for this data." % args.rms_low)
    raise SystemExit(0)

print("\n" + "=" * 78)
print("RMSA rule comparison over %d recordings (threshold %.2f)" % (n_done, args.threshold))
print("=" * 78)
print("%-10s %8s %8s %9s %9s %9s %7s" %
      ("rule", "kept%", "kept med", "sens_ovlp", "FA_ep/h", "score", "dead"))
for rule in RULES:
    a = acc[rule]
    if not a["kept"]:
        print("%-10s  <no channels available for this rule>" % rule)
        continue
    print("%-10s %8.1f %8.1f %9.3f %9.1f %9.1f %7d" % (
        rule, 100 * np.mean(a["kept"]), 100 * np.median(a["kept"]),
        np.nanmean(a["sens"]), np.nanmean(a["fa"]), np.nanmean(a["score"]), a["dead"]))

print("\nkept%%   : mean fraction of epochs the filter leaves as candidate alarms")
print("dead    : recordings where the filter suppressed EVERYTHING (sensitivity forced to 0)")
print("sens/FA/score are per-recording means, computed exactly as main_func reports them.")
