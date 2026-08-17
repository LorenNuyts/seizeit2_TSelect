"""Consensus ranking (rho_c) with the decision threshold chosen on the *validation* data.

Runs the stored models over each fold's validation subjects, picks the threshold that
maximises the reported score there -- per fold and separately per arm -- and reads the
stored *test* curve at that threshold. No model is trained and no test prediction is
recomputed; the test numbers come from the same ``*__all_results.pkl`` curves the paper
was built from.

Why this exists
---------------
``analysis/consensus_reanalysis.py`` reads the score at the fixed threshold 0.5, which
``pipeline_audit.md`` (finding 2) shows sits deep in the saturated region of these models:
per-fold scores there run tens to hundreds of points below what the same predictions
achieve between 0.6 and 0.8, and the size of that saturation gap differs between the two
arms. The per-fold differences rho sums therefore partly encode which arm saturates less at
0.5 rather than which channel set detects more seizures.

Choosing the threshold on the *test* fold would answer that, but it is not a reportable
number and is not done anywhere here. This script fits the threshold on each fold's
validation subjects instead. Both arms share identical folds and the validation subjects
are disjoint from the test subjects (both verified before predicting), so ``th*_{k,arm}``
carries no information about the test fold.

Two phases
----------
``--phase predict`` (needs the cluster: models, raw recordings and a GPU)
    Predicts each fold's validation recordings with that fold's stored model, writes them
    under ``<save_dir>/predictions_validation/<name>/fold_i/`` -- a **separate tree from
    the test predictions, which are never touched** -- then reduces them to one 51-point
    score curve per fold and arm and stores those curves in a small pickle. Resumable:
    recordings that already have a prediction file are skipped, and curves for folds
    computed in an earlier run are merged rather than overwritten. Use ``--folds`` and
    ``--arm`` to split the work over several jobs.

``--phase rho`` (runs anywhere the curves pickle is)
    Picks ``th*_{k,arm}`` from the validation curves, reads the stored test curves there,
    and recomputes rho, its ranking and the leave-one-fold-out stability -- reported next
    to the published threshold-0.5 statistic.

``--phase all`` (default) does both.

Outputs
-------
``analysis/results/consensus_val_threshold/``
  * ``val_threshold_curves.pkl``      - validation and test curves, per fold and arm
  * ``consensus_val_threshold.md``    - the report

Usage::

    # on the cluster, one job per arm (or per fold group)
    python -m analysis.consensus_val_threshold --phase predict --arm selection --gpu 0
    python -m analysis.consensus_val_threshold --phase predict --arm baseline --gpu 1

    # anywhere, once the pickle is available
    python -m analysis.consensus_val_threshold --phase rho
"""

import argparse
import os
import pickle
import random
import sys
from typing import Dict, List, Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Entry-point seeding, in the order the other entry points use it: random, numpy,
# net.key_generator, then tensorflow -- which is imported lazily inside the prediction
# phase, after --gpu has set CUDA_VISIBLE_DEVICES.
random_seed = 1
random.seed(random_seed)

import numpy as np  # noqa: E402

np.random.seed(random_seed)

from analysis.consensus_reanalysis import (  # noqa: E402
    BASELINE_NAME,
    FOCUS_COMBINATION,
    N_FOLDS,
    PAPER_CONSENSUS,
    PAPER_CONSENSUS_RHO_ROUNDED,
    PAPER_N_CANDIDATES,
    RESULTS_DIR,
    SELECTION_NAME,
    candidate_set,
    fmt,
    rank_combinations,
    rank_of,
)

ARMS = {"selection": SELECTION_NAME, "baseline": BASELINE_NAME}
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "results", "consensus_val_threshold")
CURVES_FILENAME = "val_threshold_curves.pkl"
PRED_FS = 1  # predictions are one per second, as in net/main_func.predict
PLATEAU_TOLERANCE = 1.0  # score points; how flat a curve is around its maximum


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #

def plateau_width(curve: np.ndarray, tolerance: float = PLATEAU_TOLERANCE) -> int:
    """How many thresholds lie within ``tolerance`` score points of a curve's maximum.

    A wide plateau means the threshold picked inside it is arbitrary.
    """
    best = np.nanmax(curve)
    return int(np.sum(curve >= best - tolerance))


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged, so Spearman is defined on tied differences."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


# --------------------------------------------------------------------------- #
# Stored artefacts
# --------------------------------------------------------------------------- #

def load_results(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, name + "__all_results.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def prepare_config(results: dict):
    """The config that produced the stored curves, with remote paths localised if needed.

    Mirrors the rewrite in ``net/main_func.predict``: a run trained on the cluster stores
    ``/cw/...`` paths, which have to be mapped when the same artefacts are read elsewhere.
    """
    from utility.constants import Paths

    config = results["config"]
    here = os.path.dirname(os.path.realpath(__file__))
    if "dtai" in config.save_dir and "dtai" not in here:
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
    if not hasattr(config, "test_batch_size"):
        config.test_batch_size = 6 * 60
    return config


def check_arms_comparable(selection: dict, baseline: dict) -> None:
    """The paired difference is only meaningful on identical folds and threshold grids."""
    if selection["thresholds"] != baseline["thresholds"]:
        raise ValueError("The two arms use different threshold grids.")
    for fold_i in range(N_FOLDS):
        for split in ("validation", "test"):
            a = sorted(map(str, selection["config"].folds[fold_i][split]))
            b = sorted(map(str, baseline["config"].folds[fold_i][split]))
            if a != b:
                raise ValueError(f"Fold {fold_i}: {split} subjects differ between the arms.")
        overlap = (set(map(str, selection["config"].folds[fold_i]["validation"]))
                   & set(map(str, selection["config"].folds[fold_i]["test"])))
        if overlap:
            raise ValueError(f"Fold {fold_i}: validation and test subjects overlap: {overlap}")
    if selection.get("rmsa_filtering") != baseline.get("rmsa_filtering"):
        raise ValueError("The two arms were evaluated with different RMSA filtering.")


# --------------------------------------------------------------------------- #
# Phase 1: validation predictions and curves (cluster)
# --------------------------------------------------------------------------- #

def validation_predictions_folder(config, name: str, fold_i: int) -> str:
    """Deliberately *not* ``get_path_predictions_folder``: the test predictions of these
    runs are the paper's artefacts and must not be written over."""
    return os.path.join(config.save_dir, "predictions_validation", name, f"fold_{fold_i}")


def predict_validation_fold(config, name: str, fold_i: int) -> int:
    """Predict a fold's validation recordings with that fold's stored model.

    A transcription of ``net/main_func.predict_per_fold`` with the test subjects replaced
    by the validation subjects and the output folder replaced; everything that touches the
    model, the segments and the generator is kept identical, so the resulting probabilities
    are on the same footing as the stored test predictions.
    """
    import gc

    import h5py
    from tensorflow.keras import backend as K
    from tqdm import tqdm

    from net.MiniRocket_LR import MiniRocketLR
    from net.generator_ds import SequentialGenerator
    from net.key_generator import generate_data_keys_sequential
    from net.routines import predict_net
    from utility import get_recs_list
    from utility.constants import Keys
    from utility.paths import get_path_model, get_path_model_weights

    validation_subjects = config.folds[fold_i]["validation"]
    K.clear_session()
    gc.collect()
    config.reload_CH(fold=fold_i)
    recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)

    model_save_path = get_path_model(config, name, fold_i)
    model_weights_path = get_path_model_weights(model_save_path, name)
    # Fail here, with the path, rather than deep inside Keras after the recording list has
    # already been walked -- this script is meant to run where nobody is watching.
    required = (model_save_path if config.model.lower() == Keys.minirocketLR.lower()
                else model_weights_path)
    if not os.path.exists(required):
        raise FileNotFoundError(
            f"Fold {fold_i} of '{name}' has no stored model at {required}. This script "
            f"predicts with the models the run left behind; it never trains. Check that "
            f"the run name and save_dir resolve to the machine you are on.")

    if config.model == "DeepConvNet":
        from net.DeepConv_Net import net
    elif config.model == "ChronoNet":
        from net.ChronoNet import net
    elif config.model == "EEGnet":
        from net.EEGnet import net
    elif config.model.lower() != Keys.minirocketLR.lower():
        raise ValueError("Model not recognized")

    if config.model.lower() == Keys.minirocketLR.lower():
        model = MiniRocketLR(model_save_path)
    else:
        model = net(config)
        K.set_image_data_format("channels_last")
        model.load_weights(model_weights_path)

    folder = validation_predictions_folder(config, name, fold_i)
    os.makedirs(folder, exist_ok=True)

    written = 0
    for rec in tqdm(recs_list, desc=f"fold {fold_i} validation"):
        out_path = os.path.join(folder, rec[0] + "__" + rec[1] + "__" + rec[2] + "__preds.h5")
        if os.path.isfile(out_path):
            continue
        segments = generate_data_keys_sequential(config, [rec], verbose=False)
        gen = SequentialGenerator(
            config, [rec], segments, batch_size=len(segments),
            channels=config.selected_channels[fold_i] if config.channel_selection else None,
            shuffle=False, verbose=False)

        config.reload_CH(fold_i)  # DO NOT REMOVE THIS (see net/main_func.predict_per_fold)

        if config.model.lower() == Keys.minirocketLR.lower():
            y_pred, y_true = model.predict(gen)
        else:
            y_pred, y_true = predict_net(gen, model_weights_path, model)

        del gen, segments
        gc.collect()

        with h5py.File(out_path, "w") as handle:
            handle.create_dataset("y_pred", data=y_pred)
            handle.create_dataset("y_true", data=y_true)
        config.reload_CH()
        written += 1

        del y_pred, y_true
        gc.collect()

    del model
    K.clear_session()
    gc.collect()
    return written


def validation_curve(config, name: str, fold_i: int, thresholds: Sequence[float],
                     rmsa_filtering: bool) -> Tuple[np.ndarray, int]:
    """One score curve for a fold's validation set, aggregated exactly like the test one.

    ``net/main_func.evaluate`` takes ``np.nanmean`` over recordings, per threshold column,
    of the per-recording metrics from ``get_results_rec_file``. The same call is reused
    here, so the validation and test curves differ only in which recordings they cover.
    """
    from tqdm import tqdm

    from net.main_func import get_results_rec_file
    from utility.constants import Metrics

    folder = validation_predictions_folder(config, name, fold_i)
    files = sorted(f for f in os.listdir(folder) if f.endswith("__preds.h5"))
    if not files:
        raise FileNotFoundError(f"No validation predictions in {folder}")

    per_recording = []
    for file in tqdm(files, desc=f"fold {fold_i} scoring"):
        metrics = get_results_rec_file(config, file, os.path.join(folder, file), fold_i,
                                       PRED_FS, list(thresholds),
                                       rmsa_filtering=rmsa_filtering)
        per_recording.append(metrics[Metrics.score])
    return np.nanmean(np.array(per_recording, dtype=float), axis=0), len(files)


def phase_predict(arms: Sequence[str], folds: Sequence[int], curves_path: str,
                  rmsa_override: bool | None, rescore: bool = False) -> dict:
    stored = {arm: load_results(ARMS[arm]) for arm in ARMS}
    check_arms_comparable(stored["selection"], stored["baseline"])
    thresholds = list(stored["selection"]["thresholds"])
    rmsa_filtering = (stored["selection"].get("rmsa_filtering", True)
                      if rmsa_override is None else rmsa_override)

    payload = load_curves(curves_path) or {
        "thresholds": thresholds,
        "rmsa_filtering": rmsa_filtering,
        "selected_channels": {k: list(v) for k, v in
                              stored["selection"]["config"].selected_channels.items()},
        "arms": {arm: {"val": {}, "test": {}, "n_val_recordings": {}} for arm in ARMS},
    }
    if payload["thresholds"] != thresholds:
        raise ValueError(f"{curves_path} was written with a different threshold grid.")
    if payload["rmsa_filtering"] != rmsa_filtering:
        raise ValueError(f"{curves_path} was written with rmsa_filtering="
                         f"{payload['rmsa_filtering']}, now asked for {rmsa_filtering}.")

    for arm in arms:
        name = ARMS[arm]
        config = prepare_config(stored[arm])
        for fold_i in folds:
            print(f"\n=== {arm} arm, fold {fold_i} ===")
            written = predict_validation_fold(config, name, fold_i)
            print(f"  {written} new prediction file(s)")
            # Scoring reloads the raw recordings for the RMSA mask, so it is the expensive
            # half of a resumed run. Skip it when this fold's curve is already stored and
            # no prediction changed; --rescore forces it.
            cached = payload["arms"][arm]["val"].get(fold_i)
            if cached is not None and written == 0 and not rescore:
                print("  curve already stored and no new predictions -- not re-scoring "
                      "(pass --rescore to force)")
                curve = cached
                n_files = payload["arms"][arm]["n_val_recordings"].get(fold_i, -1)
            else:
                curve, n_files = validation_curve(config, name, fold_i, thresholds,
                                                  rmsa_filtering)
            payload["arms"][arm]["val"][fold_i] = curve
            payload["arms"][arm]["n_val_recordings"][fold_i] = n_files
            payload["arms"][arm]["test"][fold_i] = np.asarray(
                stored[arm]["score"][fold_i], dtype=float)
            best = argmax_plateau(curve)
            print(f"  validation optimum: th = {thresholds[best]:.2f}, "
                  f"score = {curve[best]:.2f} over {n_files} recordings")
            save_curves(payload, curves_path)  # checkpoint after every fold

    return payload


# --------------------------------------------------------------------------- #
# Curves pickle
# --------------------------------------------------------------------------- #

def load_curves(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_curves(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)


def completed_folds(payload: dict) -> List[int]:
    return sorted(set(payload["arms"]["selection"]["val"])
                  & set(payload["arms"]["baseline"]["val"]))


# --------------------------------------------------------------------------- #
# Phase 2: rho
# --------------------------------------------------------------------------- #

def argmax_plateau(curve: np.ndarray) -> int:
    """Index of the curve's maximum, taking the middle of a tie plateau.

    Plain ``argmax`` returns the *lowest* threshold achieving the maximum, i.e. the most
    trigger-happy end of a flat region. Since the threshold is then transferred to the test
    curve, where the edge of a plateau is exactly where the score starts to fall away, the
    centre of the plateau is the robust choice. A fold whose arm never beats the never-fires
    detector has a wide plateau of exact zeros, and this is what stops it picking threshold
    0 there.
    """
    finite = np.isfinite(curve)
    best = np.nanmax(curve)
    tied = np.where(finite & (curve >= best - 1e-9))[0]
    return int(tied[len(tied) // 2])


def policy_scores(payload: dict, policy: str, folds: Sequence[int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold (selection score, baseline score, selection threshold, baseline threshold).

    ``validation``  threshold from the validation curve, per fold and per arm; the score is
                    read off the stored test curve there.
    ``th05``        the fixed threshold 0.5, i.e. the published statistic.

    No policy here reads the test curve to *choose* a threshold -- the test curve is only
    ever evaluated at a threshold fitted elsewhere.
    """
    thresholds = np.asarray(payload["thresholds"], dtype=float)
    out = {}
    for arm in ("selection", "baseline"):
        test = np.array([payload["arms"][arm]["test"][k] for k in folds], dtype=float)
        if policy == "th05":
            index = np.full(len(folds), int(np.where(np.isclose(thresholds, 0.5))[0][0]))
        elif policy == "validation":
            val = np.array([payload["arms"][arm]["val"][k] for k in folds], dtype=float)
            index = np.array([argmax_plateau(curve) for curve in val])
        else:
            raise ValueError(f"Unknown policy {policy!r}")
        out[arm] = (test[np.arange(len(folds)), index], thresholds[index])
    return out["selection"][0], out["baseline"][0], out["selection"][1], out["baseline"][1]


def analyse(payload: dict, policy: str, folds: Sequence[int],
            selected_channels: Dict[int, List[str]]) -> dict:
    scores, scores_base, th_sel, th_base = policy_scores(payload, policy, folds)
    differences_by_fold = {k: float(scores[i] - scores_base[i]) for i, k in enumerate(folds)}
    # rank_combinations indexes ``differences`` by fold number, so pad to the full range.
    differences = [differences_by_fold.get(k, 0.0) for k in range(N_FOLDS)]
    candidates = candidate_set(selected_channels, folds)
    # Denominator K = N_FOLDS, exactly as in the published definition, so a fold's
    # contribution does not change as more folds arrive from the cluster: a partial run is
    # the published rho with the missing folds contributing zero, not a rescaled statistic.
    # Leave-one-fold-out keeps the K-1 convention of analysis/consensus_reanalysis.
    ranking = rank_combinations(candidates, selected_channels, differences, folds, N_FOLDS)

    lofo = {}
    for excluded in folds:
        kept = [k for k in folds if k != excluded]
        lofo[excluded] = rank_combinations(candidate_set(selected_channels, kept),
                                           selected_channels, differences, kept,
                                           N_FOLDS - 1)[0]
    return {
        "policy": policy,
        "scores": scores,
        "scores_base": scores_base,
        "th_sel": th_sel,
        "th_base": th_base,
        "differences": differences,
        "candidates": candidates,
        "ranking": ranking,
        "lofo": lofo,
    }


def markdown_report(payload: dict, folds: Sequence[int],
                    selected_channels: Dict[int, List[str]], results: Dict[str, dict]) -> str:
    thresholds = np.asarray(payload["thresholds"], dtype=float)
    reference = results["th05"]
    validation = results["validation"]
    partial = len(folds) != N_FOLDS

    out = [
        "# Consensus ranking with the threshold chosen on validation data",
        "",
        f"Configuration: `{SELECTION_NAME}` versus the all-channel baseline "
        f"`{BASELINE_NAME}`. Folds included: {list(folds)}"
        + (f" (**partial run**: rho keeps the published denominator K = {N_FOLDS}, so the "
           f"missing folds contribute zero. Each fold's contribution is therefore final as "
           f"more folds arrive, but the ranking is provisional -- a combination occurring "
           f"only in a missing fold sits at rho = 0)" if partial else
           f", the full K = {N_FOLDS}") + ".",
        "",
        "For each fold and **each arm separately**, the decision threshold is the argmax of "
        "that arm's score curve on the fold's **validation** subjects; the score reported is "
        "the stored **test** curve read at that threshold. Validation and test subjects are "
        "disjoint and both arms share identical folds (checked before predicting), so the "
        "threshold carries no test-fold information.",
        "",
        f"RMSA filtering: {payload['rmsa_filtering']} (as stored). Validation recordings per "
        f"fold: "
        + ", ".join(f"{k}: {payload['arms']['selection']['n_val_recordings'].get(k, '?')}"
                    for k in folds) + ".",
        "",
    ]

    if not partial:
        reproduced = (len(reference["candidates"]) == PAPER_N_CANDIDATES
                      and reference["ranking"][0][0] == PAPER_CONSENSUS
                      and round(reference["ranking"][0][1]) == PAPER_CONSENSUS_RHO_ROUNDED)
        out += [f"Reproduction gate: at threshold 0.5 this pipeline returns |C| = "
                f"{len(reference['candidates'])} (paper: {PAPER_N_CANDIDATES}), argmax "
                f"{{{fmt(reference['ranking'][0][0])}}} (paper: {{{fmt(PAPER_CONSENSUS)}}}), "
                f"rho = {reference['ranking'][0][1]:.4f} (paper: "
                f"{PAPER_CONSENSUS_RHO_ROUNDED}). "
                + ("Reproduced exactly." if reproduced else
                   "**Mismatch -- inspect before using anything below.**"), ""]

    # ---- where validation puts the threshold ------------------------------- #
    out += ["## The threshold each arm gets, and where it lands", "",
            "`th val` is the threshold fitted on the fold's validation subjects; `val score` "
            "is the score it achieves there; `test @val th` is the stored test curve read at "
            "that threshold, which is the number rho uses. `test @0.5` is the same test curve "
            f"at the published threshold, for comparison. `plateau` counts validation "
            f"thresholds within {PLATEAU_TOLERANCE:g} score point of the validation optimum -- "
            "a wide plateau means the exact threshold inside it is arbitrary, and the middle "
            "of the plateau is what gets picked.",
            "",
            "| k | arm | th val | val score | test @val th | test @0.5 | change | plateau |",
            "|---|---|---|---|---|---|---|---|"]
    index_05 = int(np.where(np.isclose(thresholds, 0.5))[0][0])
    for k in folds:
        for arm in ("selection", "baseline"):
            val_curve = payload["arms"][arm]["val"][k]
            test_curve = payload["arms"][arm]["test"][k]
            i_val = argmax_plateau(val_curve)
            out.append(
                f"| {k} | {arm} | {thresholds[i_val]:.2f} | {val_curve[i_val]:.2f} | "
                f"{test_curve[i_val]:.2f} | {test_curve[index_05]:.2f} | "
                f"{test_curve[i_val] - test_curve[index_05]:+.2f} | "
                f"{plateau_width(val_curve)} |")
    out.append("")

    # ---- per-policy -------------------------------------------------------- #
    for policy, entry in results.items():
        ranking = entry["ranking"]
        differences = entry["differences"]
        focus = next((row for row in ranking if row[0] == FOCUS_COMBINATION), None)
        out += [f"## Policy `{policy}`", "",
                {"th05": "The published statistic, at the fixed threshold 0.5.",
                 "validation": "Threshold fitted on the fold's validation subjects, per "
                               "fold and per arm."}[policy],
                "",
                "| k | \\|S_k\\| | th sel | th base | score base | score sel | difference | "
                "difference @0.5 | S_k |",
                "|---|---|---|---|---|---|---|---|---|"]
        for i, k in enumerate(folds):
            out.append(
                f"| {k} | {len(selected_channels[k])} | {entry['th_sel'][i]:.2f} | "
                f"{entry['th_base'][i]:.2f} | {entry['scores_base'][i]:.2f} | "
                f"{entry['scores'][i]:.2f} | {differences[k]:+.2f} | "
                f"{reference['differences'][k]:+.2f} | "
                f"{fmt(frozenset(selected_channels[k]))} |")
        n_same = sum(1 for row in entry["lofo"].values() if row[0] == ranking[0][0])
        out += ["",
                f"argmax: **{{{fmt(ranking[0][0])}}}**, rho = {ranking[0][1]:.4f}, from "
                f"fold(s) {ranking[0][2]}. Selection beats the baseline in "
                f"**{sum(1 for k in folds if differences[k] > 0)} of {len(folds)} folds**; "
                f"{sum(1 for row in ranking if row[1] > 0)} of {len(ranking)} combinations "
                f"have rho > 0; leave-one-fold-out reproduces the argmax "
                f"{n_same}/{len(folds)} times.", ""]
        if focus is not None:
            out += [f"{{{fmt(FOCUS_COMBINATION)}}}: rho = {focus[1]:.4f}, rank "
                    f"**{rank_of(ranking, FOCUS_COMBINATION)} of {len(ranking)}**, from "
                    f"fold(s) {focus[2]}.", ""]
        out += ["| Rank | Channels | \\|c\\| | rho | Folds |", "|---|---|---|---|---|"]
        for position, (combination, value, contributing) in enumerate(ranking[:10], start=1):
            out.append(f"| {position} | {fmt(combination)} | {len(combination)} | "
                       f"{value:.4f} | {len(contributing)} |")
        out.append("")

    # ---- summary ----------------------------------------------------------- #
    score_ranks = _ranks([reference["differences"][k] for k in folds])
    out += ["## Summary", "",
            "| policy | threshold fitted on | argmax | rho | folds improved | "
            "rank of {F7, T7} | LOFO stable | Spearman vs th=0.5 |",
            "|---|---|---|---|---|---|---|---|"]
    fitted_on = {"th05": "nothing (fixed at 0.5)", "validation": "**validation subjects**"}
    for policy, entry in results.items():
        ranking = entry["ranking"]
        differences = [entry["differences"][k] for k in folds]
        n_same = sum(1 for row in entry["lofo"].values() if row[0] == ranking[0][0])
        out.append(
            f"| `{policy}` | {fitted_on[policy]} | {fmt(ranking[0][0])} | "
            f"{ranking[0][1]:.3f} | {sum(1 for d in differences if d > 0)}/{len(folds)} | "
            f"{rank_of(ranking, FOCUS_COMBINATION)} | {n_same}/{len(folds)} | "
            f"{_spearman(_ranks(differences), score_ranks):+.3f} |")

    out += ["",
            f"The validation-fitted rho is {validation['ranking'][0][1]:.2f}, against "
            f"{reference['ranking'][0][1]:.2f} at the published fixed threshold of 0.5. Both "
            f"are computed on the same stored test curves and the same folds, so the "
            f"difference between them is the threshold and nothing else.",
            "",
            "## Caveats",
            "",
            "* The validation subjects were used to pick the epoch and (in the selection "
            "arm) the channels, so a threshold fitted on them is not fitted on wholly fresh "
            "data. It is still independent of the test fold, which is what rho compares.",
            "* Curves are `np.nanmean` over recordings, per column, on both sides "
            "(`pipeline_audit.md` finding 2b) -- not the pooled challenge score.",
            "* A fold whose arm never beats the never-fires detector has a maximum of "
            "exactly 0; its difference is then minus the other arm's score, not a measured "
            "deficit.",
            "* The threshold grid is `np.linspace(0, 1, 51)`; a wide validation plateau means "
            "the choice inside it is arbitrary.",
            "* The channel selection is the stored one, made with the pre-correction "
            "selector metric. Nothing here re-selects channels.",
            "",
            "## Reproducing",
            "",
            "```",
            "python -m analysis.consensus_val_threshold --phase predict   # cluster",
            "python -m analysis.consensus_val_threshold --phase rho       # anywhere",
            "```",
            ""]
    return "\n".join(out)


def phase_rho(payload: dict, folds: Sequence[int]) -> None:
    selected_channels = {int(k): list(v) for k, v in payload["selected_channels"].items()}
    results = {policy: analyse(payload, policy, folds, selected_channels)
               for policy in ("th05", "validation")}

    thresholds = np.asarray(payload["thresholds"], dtype=float)
    index_05 = int(np.where(np.isclose(thresholds, 0.5))[0][0])
    print("\n== Validation-chosen thresholds ==")
    print(f"{'k':>2} {'arm':>10} {'th val':>7} {'val score':>10} {'test@val':>9} "
          f"{'test@0.5':>9} {'plateau':>8}")
    for k in folds:
        for arm in ("selection", "baseline"):
            val_curve, test_curve = payload["arms"][arm]["val"][k], payload["arms"][arm]["test"][k]
            i_val = argmax_plateau(val_curve)
            print(f"{k:>2} {arm:>10} {thresholds[i_val]:>7.2f} {val_curve[i_val]:>10.2f} "
                  f"{test_curve[i_val]:>9.2f} {test_curve[index_05]:>9.2f} "
                  f"{plateau_width(val_curve):>8}")

    for policy, entry in results.items():
        ranking = entry["ranking"]
        differences = [entry["differences"][k] for k in folds]
        n_same = sum(1 for row in entry["lofo"].values() if row[0] == ranking[0][0])
        print(f"\n== {policy} ==")
        for i, k in enumerate(folds):
            print(f"  fold {k}: th {entry['th_sel'][i]:.2f}/{entry['th_base'][i]:.2f}  "
                  f"base {entry['scores_base'][i]:8.2f}  sel {entry['scores'][i]:8.2f}  "
                  f"diff {entry['differences'][k]:+8.2f}")
        print(f"  argmax {{{fmt(ranking[0][0])}}}  rho={ranking[0][1]:.4f}  "
              f"({sum(1 for d in differences if d > 0)}/{len(folds)} folds improved, "
              f"LOFO {n_same}/{len(folds)}), {{{fmt(FOCUS_COMBINATION)}}} at rank "
              f"{rank_of(ranking, FOCUS_COMBINATION)} of {len(ranking)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "consensus_val_threshold.md")
    with open(path, "w") as handle:
        handle.write(markdown_report(payload, folds, selected_channels, results))
    print(f"\nWrote {path}")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["predict", "rho", "all"], default="all")
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="Arms to predict (default: both). Split across jobs if needed.")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)),
                        help="Folds to predict (default: all).")
    parser.add_argument("--gpu", type=int, default=0, help="Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--curves", default=os.path.join(OUTPUT_DIR, CURVES_FILENAME),
                        help="Where the validation/test curves are cached.")
    parser.add_argument("--no-rmsa", dest="rmsa", action="store_false", default=None,
                        help="Force RMSA filtering off (default: whatever the runs stored).")
    parser.add_argument("--rescore", action="store_true",
                        help="Recompute stored validation curves even when no prediction "
                             "changed (scoring reloads raw recordings for the RMSA mask).")
    args = parser.parse_args()

    if args.phase in ("predict", "all"):
        # Before any tensorflow import, as in the other entry points.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from net import key_generator
        key_generator.random.seed(random_seed)
        import tensorflow as tf
        tf.random.set_seed(random_seed)
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                print(error)

        payload = phase_predict(args.arm, args.folds, args.curves, args.rmsa, args.rescore)
        print(f"\nCurves written to {args.curves}")
    else:
        payload = load_curves(args.curves)
        if payload is None:
            raise SystemExit(f"{args.curves} not found -- run --phase predict on the cluster "
                             f"first, then copy the pickle here.")

    if args.phase in ("rho", "all"):
        folds = completed_folds(payload)
        if not folds:
            raise SystemExit("No fold has validation curves for both arms yet.")
        missing = sorted(set(range(N_FOLDS)) - set(folds))
        if missing:
            print(f"WARNING: folds {missing} are missing for at least one arm. rho keeps the "
                  f"published denominator K = {N_FOLDS}, so those folds contribute zero: the "
                  f"values below are final per fold but the ranking is provisional until all "
                  f"{N_FOLDS} are present.")
        phase_rho(payload, folds)


if __name__ == "__main__":
    main()
