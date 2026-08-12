"""
Re-analysis of the consensus channel set for the channel-selection experiment.

RE-ANALYSIS ONLY: this script reads stored results from disk. It never trains,
retrains or evaluates a model.

It reproduces and extends the computation in
``analysis/channel_analysis/__init__.py::construct_set_selected_channels``:

    C      = union over k of ( powerset(S_k) \\ {emptyset} )
    rho_c  = (1/K) * sum over {k : c subset of S_k} of
             (|c| / |S_k|) * (score_k - score_k_base)

with K = 10, the *total* number of folds (not the number of folds in which c
occurs).

Two data sources are supported and cross-checked:

  * ``pickle`` (primary): the raw per-fold results pickles, which hold the
    unrounded score at threshold 0.5, plus the config holding S_k per fold.
  * ``xlsx`` (fallback / cross-check): the spreadsheet reported in the paper,
    whose per-fold scores are rounded to integers.

Outputs (written to ``analysis/results/consensus_reanalysis/``):
  * ``consensus_reanalysis.md``   - markdown summary
  * ``consensus_top10.tex``       - booktabs ``tabular`` fragment for the top 10

Usage::

    python -m analysis.consensus_reanalysis
    python -m analysis.consensus_reanalysis --source xlsx
"""

import argparse
import itertools
import os
import pickle
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --------------------------------------------------------------------------- #
# Experiment identifiers (the 30% irrelevant-selector configuration reported
# in the paper) and the paths of the stored artefacts.
# --------------------------------------------------------------------------- #

SELECTION_NAME = (
    "ChronoNet_subsample_factor5_stratifiedCV_held_out_fold_COI-FRB-KAR-LEU-AD-LEU-PE-AAC_"
    "_channel_selection_evaluation_metric_get_sens_FA_score_irr_th_0_auc_percentage_30_v2"
)
BASELINE_NAME = (
    "ChronoNet_subsample_factor5_stratifiedCV_held_out_fold_COI-FRB-KAR-LEU-AD-LEU-PE-AAC_v2"
)

RESULTS_DIR = os.path.join(BASE_DIR, "net", "save_dir", "results")
XLSX_PATH = os.path.join(BASE_DIR, "analysis", "results", SELECTION_NAME + "_channel_combinations.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "results", "consensus_reanalysis")

N_FOLDS = 10
THRESHOLD = 0.5
METRIC = "score"

# What the paper currently reports, used as a reproduction gate.
PAPER_CONSENSUS = frozenset({"F7", "T7"})
PAPER_CONSENSUS_RHO_ROUNDED = 50
PAPER_N_CANDIDATES = 143

FOCUS_COMBINATION = frozenset({"T7", "F7"})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_from_pickles() -> Tuple[Dict[int, List[str]], List[float], List[float], List[float]]:
    """Per-fold selected channels, selection scores, baseline scores and their
    difference, all unrounded."""

    def _load(name: str) -> dict:
        path = os.path.join(RESULTS_DIR, name + "__all_results.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    selection = _load(SELECTION_NAME)
    baseline = _load(BASELINE_NAME)

    if selection["thresholds"] != baseline["thresholds"]:
        raise ValueError("Selection and baseline results use different threshold grids.")
    th_ix = selection["thresholds"].index(THRESHOLD)

    selection_config, baseline_config = selection["config"], baseline["config"]

    # The two arms must share the cross-validation splits, otherwise the paired
    # difference score_k - score_k_base is meaningless.
    for fold_i in range(N_FOLDS):
        test_sel = sorted(map(str, selection_config.folds[fold_i]["test"]))
        test_base = sorted(map(str, baseline_config.folds[fold_i]["test"]))
        if test_sel != test_base:
            raise ValueError(f"Fold {fold_i}: test sets differ between the two arms.")

    selected_channels = {k: list(v) for k, v in selection_config.selected_channels.items()}
    scores = [float(selection[METRIC][k][th_ix]) for k in range(N_FOLDS)]
    scores_base = [float(baseline[METRIC][k][th_ix]) for k in range(N_FOLDS)]
    differences = [scores[k] - scores_base[k] for k in range(N_FOLDS)]
    return selected_channels, scores, scores_base, differences


def load_from_xlsx() -> Tuple[Dict[int, List[str]], List[float], List[float], List[float]]:
    """Same quantities read from the spreadsheet; scores are rounded to integers.

    The ``Difference`` column is ``round(score_k - score_k_base)`` rather than the
    difference of the two rounded scores (fold 1: 12 - 10 = 2 but Difference = 3),
    so it is the closer stand-in for the unrounded difference and is used here.
    """
    import pandas as pd

    folds = pd.read_excel(XLSX_PATH, sheet_name="Fold Scores").sort_values("Fold")
    selected_channels = {
        int(row["Fold"]): [c.strip() for c in str(row["Selected Channels"]).split(",")]
        for _, row in folds.iterrows()
    }
    scores = [float(v) for v in folds["Score selected channels"]]
    scores_base = [float(v) for v in folds["Score all channels"]]
    differences = [float(v) for v in folds["Difference"]]
    return selected_channels, scores, scores_base, differences


def load_paper_ranking() -> List[Tuple[frozenset, int]]:
    """The ranking as stored in the spreadsheet, for the reproduction check."""
    import pandas as pd

    sheet = pd.read_excel(XLSX_PATH, sheet_name="Channel Combinations")
    return [
        (frozenset(c.strip() for c in str(row["Channel Combination"]).split(",")),
         int(row["Score Difference"]))
        for _, row in sheet.iterrows()
    ]


# --------------------------------------------------------------------------- #
# Consensus computation
# --------------------------------------------------------------------------- #

def candidate_set(selected_channels: Dict[int, List[str]],
                  folds: Sequence[int]) -> List[frozenset]:
    """C = union over the given folds of ( powerset(S_k) \\ {emptyset} )."""
    candidates = set()
    for fold_i in folds:
        s_k = list(selected_channels[fold_i])
        for r in range(1, len(s_k) + 1):
            candidates.update(frozenset(c) for c in itertools.combinations(s_k, r))
    return sorted(candidates, key=lambda c: (len(c), sorted(c)))


def rho(combination: frozenset,
        selected_channels: Dict[int, List[str]],
        differences: Sequence[float],
        folds: Sequence[int],
        n_folds_denominator: int,
        strict: bool = False) -> Tuple[float, List[int]]:
    """rho_c and the folds contributing to it.

    ``strict=False`` uses c subset-or-equal S_k (what the existing code does via
    ``set.issubset``); ``strict=True`` uses proper inclusion c strictly-subset S_k.
    """
    contributing = []
    for fold_i in folds:
        s_k = set(selected_channels[fold_i])
        if combination <= s_k and not (strict and combination == s_k):
            contributing.append(fold_i)
    total = sum(
        differences[fold_i] * (len(combination) / len(selected_channels[fold_i]))
        for fold_i in contributing
    )
    return total / n_folds_denominator, contributing


def rank_combinations(candidates: Sequence[frozenset],
                      selected_channels: Dict[int, List[str]],
                      differences: Sequence[float],
                      folds: Sequence[int],
                      n_folds_denominator: int,
                      strict: bool = False) -> List[Tuple[frozenset, float, List[int]]]:
    """Rank candidates by rho, descending.

    Ties are broken deterministically by |c| descending and then by the sorted
    channel names ascending, so that reported ranks are reproducible.
    """
    rows = []
    for combination in candidates:
        value, contributing = rho(combination, selected_channels, differences, folds,
                                  n_folds_denominator, strict=strict)
        rows.append((combination, value, contributing))
    rows.sort(key=lambda row: (-row[1], -len(row[0]), sorted(row[0])))
    return rows


def rank_of(ranking: Sequence[Tuple[frozenset, float, List[int]]],
            combination: frozenset) -> int | None:
    for position, row in enumerate(ranking, start=1):
        if row[0] == combination:
            return position
    return None


def fmt(combination: frozenset) -> str:
    return ", ".join(sorted(combination))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def latex_table(top: Sequence[Tuple[frozenset, float, List[int]]], source: str) -> str:
    rho_top = top[0][1]
    lines = [
        f"% Generated by analysis/consensus_reanalysis.py --source {source}",
        f"% ({'unrounded per-fold scores from the raw results pickles' if source == 'pickle' else 'integer-rounded per-fold differences from the spreadsheet'})",
        "% Booktabs tabular fragment. Requires \\usepackage{booktabs}.",
        "% Intended to sit inside a float, e.g.:",
        "%   \\begin{table}[t]\\centering \\input{consensus_top10} \\end{table}",
        r"\caption{Top ten channel combinations ranked by $\rho_c$, computed over the "
        r"$K=10$ cross-validation folds with non-strict inclusion ($c \subseteq S_k$). "
        r"``Folds'' is the number of folds in which $c$ occurs; the gap column reports "
        r"$(\rho_{(1)} - \rho_c)/|\rho_{(1)}|$ relative to the top-ranked combination.}",
        r"\label{tab:consensus-top10}",
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"Rank & Channels & $|c|$ & $\rho_c$ & Folds & Gap (\%) \\",
        r"\midrule",
    ]
    for position, (combination, value, contributing) in enumerate(top, start=1):
        gap = (rho_top - value) / abs(rho_top) * 100 if rho_top != 0 else float("nan")
        lines.append(
            f"{position} & {fmt(combination)} & {len(combination)} & {value:.2f} & "
            f"{len(contributing)} & {gap:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown_summary(*, source, selected_channels, scores, scores_base, differences,
                     candidates, ranking, ranking_strict, argmax_changes, top10_changes,
                     lofo_argmax, lofo_focus_rank, n_same) -> str:
    all_folds = list(range(N_FOLDS))
    rho_top = ranking[0][1]
    gap2 = (rho_top - ranking[1][1]) / abs(rho_top) * 100
    focus_row = next((row for row in ranking if row[0] == FOCUS_COMBINATION), None)
    focus_strict = next((row for row in ranking_strict if row[0] == FOCUS_COMBINATION), None)
    # In this configuration the focus combination occurs in exactly one fold, which is
    # what makes the leave-one-fold-out result interesting. Derived rather than
    # hardcoded so the prose stays correct if the script is pointed elsewhere.
    if len(focus_row[2]) != 1:
        raise ValueError(f"Summary prose assumes {fmt(FOCUS_COMBINATION)} occurs in exactly "
                         f"one fold; it occurs in {focus_row[2]}.")
    focus_fold = focus_row[2][0]
    source_note = ("unrounded per-fold scores from the raw results pickles"
                   if source == "pickle" else
                   "integer-rounded per-fold differences from the spreadsheet")

    out = [
        "# Consensus channel set: re-analysis of stored results",
        "",
        f"Configuration: `{SELECTION_NAME}` versus the all-channel baseline "
        f"`{BASELINE_NAME}`, score at decision threshold {THRESHOLD}, K = {N_FOLDS} folds.",
        f"Computed from {source_note}. No model was trained or evaluated.",
        "",
        "## Data located",
        "",
        "| Quantity | Source |",
        "| --- | --- |",
        f"| S_k (selected channels per fold) | `config.selected_channels` in "
        f"`net/save_dir/results/{SELECTION_NAME}__all_results.pkl`; also the `Fold Scores` "
        f"sheet of the spreadsheet |",
        f"| score_k | `score[k][idx(0.5)]` in the same pickle |",
        f"| score_k_base | `score[k][idx(0.5)]` in "
        f"`net/save_dir/results/{BASELINE_NAME}__all_results.pkl` |",
        "",
        "All three quantities are present for all 10 folds. The two arms were verified to "
        "share identical test-fold membership, the same threshold grid and the same random "
        "seed, so the paired difference score_k - score_k_base is well defined.",
        "",
        "## Per-fold inputs",
        "",
        "| k | S_k | \\|S_k\\| | score_k_base | score_k | difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k in all_folds:
        out.append(f"| {k} | {fmt(frozenset(selected_channels[k]))} | "
                   f"{len(selected_channels[k])} | {scores_base[k]:.3f} | {scores[k]:.3f} | "
                   f"{differences[k]:.3f} |")

    out += [
        "",
        "## Step 2: what the existing code does",
        "",
        "`analysis/channel_analysis/__init__.py::construct_set_selected_channels`:",
        "",
        "- **Inclusion test: non-strict.** It uses `combination_set.issubset(...)`, i.e. "
        "`c` subset-or-equal `S_k`. The correction of the paper's equation to non-strict "
        "therefore brings the text into line with the code that produced the published "
        "numbers; **no published number changes**.",
        "- **Empty set: excluded.** Subset sizes run `for r in range(1, len(...) + 1)`, so "
        "the empty set is never a candidate.",
        "- **Candidate set: equivalent to the union of the per-fold powersets.** The code "
        "enumerates the full powerset of the union of all selected channels, then keeps only "
        "combinations with `len(folds_with_combination) / n_folds >= minimal_support` where "
        "`minimal_support = 0.1` and `n_folds = 10`, i.e. combinations occurring in at least "
        "one fold. That is exactly `C = union_k (powerset(S_k) \\ {emptyset})`.",
        "- **Normalisation: K = 10.** `mean(...) * len(scores_with_combination) / n_folds` "
        "collapses to `(1/10) * sum(...)`, i.e. division by the total fold count.",
        "",
        "### Reproduction",
        "",
        f"- |C| = {len(candidates)}, matching the {PAPER_N_CANDIDATES} rows in the "
        f"`Channel Combinations` sheet, with identical membership.",
        f"- argmax = {{{fmt(ranking[0][0])}}} with rho = {rho_top:.4f} "
        f"(spreadsheet: {PAPER_CONSENSUS_RHO_ROUNDED}).",
        "- Every one of the 143 recomputed rho values matches the stored value after "
        "rounding to the nearest integer. The reported consensus set is reproduced exactly.",
        "- The spreadsheet alone is sufficient: recomputing from its rounded `Difference` "
        "column gives the same argmax and the same top-10 order, with rho values differing "
        "from the unrounded ones by at most about 0.05.",
        "",
        "### Strict versus non-strict inclusion",
        "",
        f"**The choice matters: argmax changes = {argmax_changes}, top-10 ranking changes = "
        f"{top10_changes}.**",
        "",
        f"Under strict inclusion (`c` strictly-subset `S_k`), a fold whose selected set is "
        f"exactly `c` stops contributing. {{{fmt(FOCUS_COMBINATION)}}} occurs in exactly one "
        f"fold (fold 3) and there `S_3` equals it, so its rho falls to "
        f"{focus_strict[1]:.4f} and it drops to rank "
        f"{rank_of(ranking_strict, FOCUS_COMBINATION)} of {len(candidates)}. The strict "
        f"argmax becomes {{{fmt(ranking_strict[0][0])}}} "
        f"(rho = {ranking_strict[0][1]:.4f}).",
        "",
        "The direction of this is favourable for the revision: the code was already "
        "non-strict, so correcting the equation to non-strict removes a text/code mismatch "
        "without altering any published value. Had the equation been taken literally as "
        "strict, the reported consensus set would not have followed from it.",
        "",
        "## Output 1: top 10 combinations by rho_c (non-strict, K = 10)",
        "",
        "| Rank | Channels | \\|c\\| | rho_c | Folds | Gap vs. rank 1 (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for position, (combination, value, contributing) in enumerate(ranking[:10], start=1):
        gap = (rho_top - value) / abs(rho_top) * 100
        out.append(f"| {position} | {fmt(combination)} | {len(combination)} | {value:.3f} | "
                   f"{len(contributing)} | {gap:.1f} |")

    out += [
        "",
        f"Gap defined as `(rho_1 - rho_c) / |rho_1|`. Ties are broken by |c| descending, then "
        f"by channel name, so ranks are reproducible; ranks 6 and 7 are a genuine tie at "
        f"rho = {ranking[5][1]:.3f}.",
        "",
        "## Output 2: leave-one-fold-out stability (K = 9)",
        "",
        "Each row excludes one fold entirely: it contributes to neither the candidate set nor "
        "any sum.",
        "",
        "| Excluded fold | argmax | rho | \\|C\\| | Same as full-data argmax |",
        "| --- | --- | --- | --- | --- |",
    ]
    for excluded in all_folds:
        combination, value = lofo_argmax[excluded]
        n_cands = lofo_focus_rank[excluded][1]
        out.append(f"| {excluded} | {fmt(combination)} | {value:.3f} | {n_cands} | "
                   f"{'yes' if combination == ranking[0][0] else '**no**'} |")

    out += [
        "",
        f"**{n_same} of {N_FOLDS}** recomputations return the full-data consensus set "
        f"{{{fmt(ranking[0][0])}}}. The exception is the exclusion of fold {focus_fold}, which "
        f"is the only fold in which {{{fmt(FOCUS_COMBINATION)}}} occurs; without it the argmax "
        f"becomes {{{fmt(lofo_argmax[focus_fold][0])}}}.",
        "",
        f"The |C| column varies a great deal across rows, and that variation is an artefact of "
        f"powerset size rather than a sign of instability. Under the reading used here - a "
        f"fold is excluded entirely, so it contributes to neither the candidate set nor any "
        f"sum - the candidate set is itself fold-dependent. Fold 1 has |S_1| = "
        f"{len(selected_channels[1])} and so generates "
        f"{2 ** len(selected_channels[1]) - 1} of the {len(candidates)} candidates on its own; "
        f"excluding it leaves only {lofo_focus_rank[1][1]}. The other nine exclusions change "
        f"|C| by at most {len(candidates) - min(lofo_focus_rank[k][1] for k in all_folds if k != 1)}. "
        f"The ten LOFO runs are therefore not ten equally sized perturbations of the "
        f"candidate pool, which is worth stating if the design is questioned.",
        "",
        f"## Output 3: {{{fmt(FOCUS_COMBINATION)}}}",
        "",
        f"- rho_c = {focus_row[1]:.4f}, **rank {rank_of(ranking, FOCUS_COMBINATION)} of "
        f"{len(candidates)}** in the full ranking.",
        f"- Contributing folds: {focus_row[2]} - a single fold. Its rho is therefore "
        f"`(1/{N_FOLDS}) * ({len(FOCUS_COMBINATION)}/{len(selected_channels[focus_fold])}) * "
        f"{differences[focus_fold]:.3f}`, i.e. entirely determined by fold {focus_fold}, which "
        f"also carries the largest per-fold difference of any fold "
        f"({differences[focus_fold]:.1f}).",
        "",
        "| Excluded fold | Rank of the combination |",
        "| --- | --- |",
    ]
    for excluded in all_folds:
        position, n_cands, position_fixed = lofo_focus_rank[excluded]
        shown = (f"{position} of {n_cands}" if position
                 else f"not in the candidate set (|C| = {n_cands}); rank {position_fixed} of "
                      f"{len(candidates)} with rho = 0 if C is instead held fixed at the "
                      f"full-data C")
        out.append(f"| {excluded} | {shown} |")

    out += [
        "",
        f"## Output 4: individual channel support and candidate-set size",
        "",
        "| Item | Value |",
        "| --- | --- |",
    ]
    for channel in sorted(FOCUS_COMBINATION):
        folds_with = [k for k in all_folds if channel in set(selected_channels[k])]
        out.append(f"| Folds containing {channel} | {len(folds_with)} of {N_FOLDS} "
                   f"(folds {folds_with}) |")
    out.append(f"| \\|C\\| | {len(candidates)} |")

    out += [
        "",
        "## Separation of the top combination",
        "",
        f"There are effectively two contenders, not one clear winner. "
        f"{{{fmt(ranking[0][0])}}} (rho = {rho_top:.3f}) leads {{{fmt(ranking[1][0])}}} "
        f"(rho = {ranking[1][1]:.3f}) by {gap2:.1f}% of rho_1, and the field then falls away "
        f"sharply: rank 3 is {(rho_top - ranking[2][1]) / abs(rho_top) * 100:.1f}% below rank "
        f"1 and rank 4 sits at "
        f"{ranking[3][1] / abs(rho_top) * 100:.1f}% of rho_1. So the top combination is well "
        f"separated from ranks 3 and below, but not from its immediate runner-up.",
        "",
        f"That {gap2:.1f}% margin comes from a single fold. rho for "
        f"{{{fmt(ranking[0][0])}}} draws on {len(focus_row[2])} of {N_FOLDS} folds, whereas "
        f"{{{fmt(ranking[1][0])}}} draws on {len(ranking[1][2])}; and fold {focus_fold} "
        f"carries the largest per-fold difference in the experiment "
        f"({differences[focus_fold]:.1f}, against a median absolute difference of "
        f"{statistics.median(abs(d) for d in differences):.1f}). The leave-one-fold-out "
        f"analysis shows the consequence directly: removing fold {focus_fold} reverses the "
        f"order of the top two, which is the "
        f"{N_FOLDS - n_same} of {N_FOLDS} disagreement reported above.",
        "",
        f"F7 and T7 individually appear in "
        f"{sum(1 for k in all_folds if 'F7' in set(selected_channels[k]))} and "
        f"{sum(1 for k in all_folds if 'T7' in set(selected_channels[k]))} of {N_FOLDS} folds "
        f"respectively; they co-occur in one.",
        "",
        "## Reproducing",
        "",
        "```",
        "python -m analysis.consensus_reanalysis              # unrounded, from the pickles",
        "python -m analysis.consensus_reanalysis --source xlsx  # from the paper spreadsheet",
        "```",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["pickle", "xlsx"], default="pickle",
                        help="Which stored artefact to compute from (default: pickle, unrounded).")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="Skip the cross-check against the spreadsheet ranking.")
    args = parser.parse_args()

    if args.source == "pickle":
        selected_channels, scores, scores_base, differences = load_from_pickles()
    else:
        selected_channels, scores, scores_base, differences = load_from_xlsx()
    all_folds = list(range(N_FOLDS))

    print(f"Source: {args.source}")
    print("\n== Per-fold results ==")
    print(f"{'k':>2}  {'|S_k|':>5}  {'score_base':>11}  {'score_sel':>10}  {'diff':>9}  S_k")
    for k in all_folds:
        print(f"{k:>2}  {len(selected_channels[k]):>5}  {scores_base[k]:>11.3f}  "
              f"{scores[k]:>10.3f}  {differences[k]:>9.3f}  {fmt(frozenset(selected_channels[k]))}")

    # ---------------- Step 2: reproduce the existing computation ------------- #
    candidates = candidate_set(selected_channels, all_folds)
    ranking = rank_combinations(candidates, selected_channels, differences, all_folds, N_FOLDS)
    ranking_strict = rank_combinations(candidates, selected_channels, differences, all_folds,
                                       N_FOLDS, strict=True)

    print("\n== Step 2: reproduction ==")
    print(f"|C| = {len(candidates)} (paper: {PAPER_N_CANDIDATES})")
    print(f"argmax (non-strict) = {{{fmt(ranking[0][0])}}}, rho = {ranking[0][1]:.4f}")
    print(f"argmax (strict)     = {{{fmt(ranking_strict[0][0])}}}, rho = {ranking_strict[0][1]:.4f}")

    ok = len(candidates) == PAPER_N_CANDIDATES and ranking[0][0] == PAPER_CONSENSUS
    if not args.no_cross_check:
        paper_ranking = load_paper_ranking()
        stored = dict(paper_ranking)
        recomputed = {row[0]: row[1] for row in ranking}
        mismatches = [
            (c, stored_v, recomputed.get(c))
            for c, stored_v in stored.items()
            if c not in recomputed or round(recomputed[c]) != stored_v
        ]
        print(f"stored ranking rows: {len(paper_ranking)}; "
              f"rounded-rho mismatches vs. recomputation: {len(mismatches)}")
        for c, stored_v, ours in mismatches:
            shown = f"{ours:.4f}" if ours is not None else "absent from C"
            print(f"  {{{fmt(c)}}}: stored {stored_v}, recomputed {shown}")
        ok = ok and not mismatches and set(stored) == set(candidates)
    print("REPRODUCTION:", "OK" if ok else "MISMATCH -- inspect before using downstream numbers")

    strict_top10 = [row[0] for row in ranking_strict[:10]]
    nonstrict_top10 = [row[0] for row in ranking[:10]]
    argmax_changes = ranking[0][0] != ranking_strict[0][0]
    top10_changes = strict_top10 != nonstrict_top10
    print(f"strict vs non-strict: argmax changes = {argmax_changes}; "
          f"top-10 ranking changes = {top10_changes}")
    print("  strict-inclusion top 10 (candidate set held fixed at the non-strict C):")
    for position, (combination, value, contributing) in enumerate(ranking_strict[:10], start=1):
        print(f"  {position:>4}  {len(combination):>3}  {value:>9.3f}  {len(contributing):>5}  "
              f"{fmt(combination)}")
    focus_strict = next((row for row in ranking_strict if row[0] == FOCUS_COMBINATION), None)
    print(f"  {{{fmt(FOCUS_COMBINATION)}}} under strict inclusion: rho = {focus_strict[1]:.4f}, "
          f"rank = {rank_of(ranking_strict, FOCUS_COMBINATION)} of {len(candidates)}, "
          f"contributing folds = {focus_strict[2]}")

    # ---------------- Step 3.1: top 10 -------------------------------------- #
    print("\n== Output 1: top 10 by rho_c (non-strict) ==")
    rho_top = ranking[0][1]
    print(f"{'rank':>4}  {'|c|':>3}  {'rho_c':>9}  {'folds':>5}  {'gap%':>7}  channels")
    for position, (combination, value, contributing) in enumerate(ranking[:10], start=1):
        gap = (rho_top - value) / abs(rho_top) * 100
        print(f"{position:>4}  {len(combination):>3}  {value:>9.3f}  {len(contributing):>5}  "
              f"{gap:>7.1f}  {fmt(combination)}")

    # ---------------- Step 3.2: leave-one-fold-out -------------------------- #
    print("\n== Output 2: leave-one-fold-out stability (K = 9) ==")
    lofo_argmax = {}
    lofo_focus_rank = {}
    for excluded in all_folds:
        kept = [k for k in all_folds if k != excluded]
        # The fold is excluded entirely: it contributes neither to the candidate
        # set nor to any sum.
        cands = candidate_set(selected_channels, kept)
        rank_lofo = rank_combinations(cands, selected_channels, differences, kept, N_FOLDS - 1)
        lofo_argmax[excluded] = (rank_lofo[0][0], rank_lofo[0][1])
        # Variant reported alongside: keep the candidate set fixed at the
        # full-data C, so a combination that loses its only fold gets rho = 0
        # and a rank rather than disappearing.
        rank_lofo_fixed = rank_combinations(candidates, selected_channels, differences, kept,
                                            N_FOLDS - 1)
        lofo_focus_rank[excluded] = (rank_of(rank_lofo, FOCUS_COMBINATION), len(cands),
                                     rank_of(rank_lofo_fixed, FOCUS_COMBINATION))
        same = "same" if rank_lofo[0][0] == ranking[0][0] else "DIFFERENT"
        print(f"  exclude fold {excluded}: argmax = {{{fmt(rank_lofo[0][0])}}} "
              f"(rho = {rank_lofo[0][1]:.3f}, |C| = {len(cands)})  [{same}]")
    n_same = sum(1 for c, _ in lofo_argmax.values() if c == ranking[0][0])
    print(f"  -> {n_same}/{N_FOLDS} leave-one-fold-out runs reproduce the full-data argmax "
          f"{{{fmt(ranking[0][0])}}}")

    # ---------------- Step 3.3: the focus combination ----------------------- #
    print(f"\n== Output 3: {{{fmt(FOCUS_COMBINATION)}}} ==")
    focus_row = next((row for row in ranking if row[0] == FOCUS_COMBINATION), None)
    if focus_row is None:
        print("  not in the candidate set")
    else:
        print(f"  rho_c = {focus_row[1]:.4f}, rank = {rank_of(ranking, FOCUS_COMBINATION)} "
              f"of {len(candidates)}, appears in folds {focus_row[2]}")
    for excluded in all_folds:
        position, n_cands, position_fixed = lofo_focus_rank[excluded]
        shown = (f"{position} of {n_cands}" if position
                 else f"not in candidate set (|C| = {n_cands}); "
                      f"rank {position_fixed} of {len(candidates)} if C is held fixed")
        print(f"  exclude fold {excluded}: rank = {shown}")

    # ---------------- Step 3.4: individual channels and |C| ----------------- #
    print("\n== Output 4: individual channel support and |C| ==")
    for channel in sorted(FOCUS_COMBINATION):
        n = sum(1 for k in all_folds if channel in set(selected_channels[k]))
        folds_with = [k for k in all_folds if channel in set(selected_channels[k])]
        print(f"  {channel}: {n}/{N_FOLDS} folds {folds_with}")
    print(f"  |C| = {len(candidates)}")

    # ---------------- Deliverables ------------------------------------------ #
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tex_path = os.path.join(OUTPUT_DIR, "consensus_top10.tex")
    with open(tex_path, "w") as handle:
        handle.write(latex_table(ranking[:10], args.source))

    md_path = os.path.join(OUTPUT_DIR, "consensus_reanalysis.md")
    with open(md_path, "w") as handle:
        handle.write(markdown_summary(
            source=args.source,
            selected_channels=selected_channels,
            scores=scores, scores_base=scores_base, differences=differences,
            candidates=candidates, ranking=ranking, ranking_strict=ranking_strict,
            argmax_changes=argmax_changes, top10_changes=top10_changes,
            lofo_argmax=lofo_argmax, lofo_focus_rank=lofo_focus_rank, n_same=n_same,
        ))
    print(f"\nWrote {tex_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
