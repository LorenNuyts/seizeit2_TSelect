import argparse
import os
import shutil
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.channel_analysis.file_management import download_remote_configs, download_remote_results
from analysis.utils import find_longest_common_substring
from utility.constants import evaluation_metrics, parse_location, Locations, Keys
from net.DL_config import get_base_config, get_channel_selection_config, Config
from utility.paths import get_path_results
from utility.stats import Results

base_dir = os.path.dirname(os.path.realpath(__file__))

def get_unique_config_names(configs):
    names = [config.get_name() for config in configs]
    split_names = [name.split('_') for name in names]  # You can change the delimiter as needed
    num_parts = max(len(parts) for parts in split_names)

    unique_parts = []
    for i in range(len(configs)):
        unique = []
        for j in range(num_parts):
            parts_at_j = [split[j] if j < len(split) else "" for split in split_names]
            if len(set(parts_at_j)) > 1:
                if j < len(split_names[i]):
                    unique.append(split_names[i][j])
        unique_parts.append('_'.join(unique) if unique else names[i])
    return dict(zip(names, unique_parts))

def violin_plot(configs: list, metric: str, threshold: float, output_path: str):
    data = []
    full_to_short_names = get_unique_config_names(configs)

    for config in configs:
        results_path = os.path.join(base_dir, "..", get_path_results(config, config.get_name()))
        results = Results(config)
        print("Now handling: ", results_path)
        if os.path.exists(results_path):
            results.load_results(results_path)
        else:
            print(f"Results not found for {config.get_name()}")
            continue

        all_values = getattr(results, metric, None)
        if all_values is None:
            raise ValueError(f"Metric {metric} not found in results for {config.get_name()}")
        else:
            threshold_index = results.thresholds.index(threshold)
            values_threshold =list(zip(*all_values))[threshold_index]
            for val in values_threshold:
                if np.isnan(val):
                    continue
                label = config.pretty_name if config.pretty_name else full_to_short_names[config.get_name()]
                data.append({
                    "Configuration": label,
                    "Value": val
                })

            # Print how many values were NaN
            print(f"Number of NaN values for {config.get_name()} with threshold {threshold}: ",
                  len([val for val in values_threshold if np.isnan(val)]),
                  f"out of {len(values_threshold)} values")

            df = pd.DataFrame(data)

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x="Configuration", y="Value", inner="box", cut=0)
            plt.xlabel("")
            plt.ylabel(metric.replace("_", " ").title())
            plt.tight_layout()

            if output_path:
                plot_path = os.path.splitext(output_path)[0] + (f"_{metric.replace('.', '_')}"
                                                                f"_th{str(threshold).replace('.', '')}"
                                                                f"_cut.png")
                if not os.path.exists(os.path.dirname(plot_path)):
                    os.makedirs(os.path.dirname(plot_path))
                plt.savefig(plot_path)
                print(f"Violin plot saved to {plot_path}")

            plt.show()

            plt.close()

def plot_varying_thresholds(configs: List[Config], metrics: List[str], output_path: str, rmsa_filtering: bool = True,
                            split_localization: bool = False):
    # included_folds = [1,3,4,5,7,8,9]
    full_to_short_names = get_unique_config_names(configs)
    if split_localization:
        all_lateralizations = ['left', 'right', 'unknown', 'no_seizures', 'bilateral']  # , 'mixed']
    else:
        all_lateralizations = [None]


    fah_epoch_included = "fah_epoch" in metrics
    if not fah_epoch_included:
        metrics.append("fah_epoch")

    for lat in all_lateralizations:
        data = defaultdict(dict)
        thresholds = None
        for metric in metrics:
            for config in configs:
                results_path = os.path.join(base_dir, "..", get_path_results(config, config.get_name() + (f"_{lat}" if lat else "")))
                if not rmsa_filtering:
                    results_path = results_path.replace('.pkl', '_noRMSA.pkl')
                results = Results(config)
                print("Now handling: ", results_path)
                if os.path.exists(results_path):
                    results.load_results(results_path)
                else:
                    print(f"Results not found for {config.get_name()}, downloading...")
                    download_remote_configs([config], local_base_dir=config.save_dir)
                    download_remote_results([config], local_base_dir=config.save_dir, rmsa_filtering=rmsa_filtering,
                                            lateralization=lat)
                    results.load_results(results_path)

                # included_values = [v for i, v in enumerate(getattr(results, metric)) if i in included_folds]
                # setattr(results, metric, included_values)

                average_values = getattr(results, f"average_{metric}_all_thresholds", None)
                if average_values is None:
                    raise ValueError(f"Metric {metric} not found in results for {config.get_name()}")

                std_values = getattr(results, f"std_{metric}_all_thresholds", None)
                if std_values is None:
                    raise ValueError(f"Standard deviation for metric {metric} not found in results for {config.get_name()}")

                if thresholds is None:
                    thresholds = results.thresholds

                label = config.pretty_name if config.pretty_name else full_to_short_names[config.get_name()]
                data[metric][label] = {
                    "average": average_values,
                    "std": std_values
                }

        # Lower bound = highest threshold where everything is predicted as a seizure -> max fah
        # Upper bound = lowest threshold where no seizure is predicted -> fah = 0
        lower_bound = len(thresholds) - 1 # Last threshold because the lower bound should be the lowest one of all configurations
        upper_bound = 0 # First threshold because the upper bound should be the highest one of all configurations
        for label, values in data["fah_epoch"].items():
            max_fah = max(values["average"])
            lower_bound_i = len(thresholds) - 1 - values["average"][::-1].index(max_fah)
            lower_bound = min(lower_bound, lower_bound_i)
            upper_bound_i = values["average"].index(0.0)
            if upper_bound_i is None:
                upper_bound_i = len(thresholds) - 1
            upper_bound = max(upper_bound, upper_bound_i)

        thresholds = thresholds[lower_bound:upper_bound]

        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        fontsize = 20
        # plt.rcParams.update({'font.size': 30})

        # # Custom colors (normalized RGB values)
        # colors = [(78 / 255, 167 / 255, 46 / 255),  # RGB(78, 167, 46)
        #           (15 / 255, 158 / 255, 213 / 255),  # RGB(15, 158, 213)
        #           (112 / 255, 48 / 255, 160 / 255),  # RGB(112, 48, 160)
        #           (153 / 255, 33 / 255, 9 / 255),  # RGB(153, 33, 9)]
        #           ]

        for metric in metrics:
            if metric == "fah_epoch" and not fah_epoch_included:
                continue

            plt.figure(figsize=(10, 6))
            for i, (label, values) in enumerate(data[metric].items()):
                average_values = values["average"][lower_bound:upper_bound]
                std_values = values["std"][lower_bound:upper_bound]
                plt.plot(thresholds, average_values, label=label, linewidth=3)
                plt.fill_between(thresholds,
                                 np.array(average_values) - np.array(std_values),
                                 np.array(average_values) + np.array(std_values),
                                 alpha=0.2)

            plt.xlabel("Threshold", fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.ylabel(pretty_print_metrics[metric], fontsize=fontsize)
            plt.yticks(fontsize=fontsize-2)
            plt.axvline(x=0.5, linestyle='--', color='black')
            if lat:
                plt.title(f"Lateralization: {lat.title()}", fontsize=fontsize + 2)
            # plt.title(configs[0].cross_validation.replace("_", " ").title(), fontsize=16)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()

            if output_path:
                suffix = configs[0].cross_validation
                if not rmsa_filtering:
                    suffix += "_noRMSA"
                if lat:
                    suffix += f"_{lat}"
                # suffix = configs[0].cross_validation + "_no_heldout"
                # suffix = configs[0].cross_validation + "_heldout_only"
                plot_path = os.path.splitext(output_path)[0] + f"_{metric.replace('.', '_')}_{suffix}.png"
                if not os.path.exists(os.path.dirname(plot_path)):
                    os.makedirs(os.path.dirname(plot_path))
                plt.savefig(plot_path)

            plt.show()
            plt.close()


if __name__ == '__main__':
    allowed_metrics = {'f1_ovlp', 'fah_ovlp', 'fah_epoch', 'prec_ovlp', 'sens_ovlp', 'score',
                       'nb_channels', 'selection_time', 'train_time', 'total_time', 'all'}
    pretty_print_metrics = {'f1_ovlp': 'F1 (overlap)', 'fah_ovlp': 'False alarm rate (overlap)',
                            'fah_epoch': 'False alarm rate (epoch)',
                            'prec_ovlp': 'Precision (overlap)', 'sens_ovlp': 'Sensitivity (overlap)', 'score': 'Score',}
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices={'violin', 'thresholds'},)
    parser.add_argument("metric", type=str, choices=allowed_metrics)
    parser.add_argument('--locations', nargs='+', type=parse_location,
                        default=[parse_location(l) for l in Locations.all_keys()],
                        help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
                             f"Defaults to all locations."
    )
    parser.add_argument("--threshold", type=float, default=0.5, nargs="?")
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    parser.add_argument("--no_rmsa", action="store_true")
    args = parser.parse_args()
    task = args.task
    metric_ = args.metric
    threshold_ = args.threshold
    suffix_ = args.suffix
    unique_locations = sorted(list(dict.fromkeys(args.locations)))

    # if metric_ not in allowed_metrics:
    #     raise ValueError(f"Metric {metric_} is not allowed. Choose from {allowed_metrics}")
    configs_stratified_old = [
        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        held_out_fold=True, pretty_name="Baseline (old)",
                        version_experiments=None),
        # get_base_config(base_dir, unique_locations, suffix="rerun", CV=Keys.stratified,
        #                 held_out_fold=True, pretty_name="Baseline"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'], CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=-100)"),
        get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'],
                                     irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                     held_out_fold=True, pretty_name="Channel Selection (old)",
                                     version_experiments=None),
    ]


    configs_stratified = [
        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        held_out_fold=True, pretty_name="Baseline (all channels)"),

        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified, included_channels="no_wearables",
                        held_out_fold=True, pretty_name="Baseline (full-scalp EEG)"),

        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        included_channels="CROSStop",
                        held_out_fold=True, pretty_name="Baseline (CROSStop SD)"),
        # get_base_config(base_dir, unique_locations, suffix="rerun", CV=Keys.stratified,
        #                 held_out_fold=True, pretty_name="Baseline"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'], CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=-100)"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0, CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (60%)"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0, irrelevant_selector_percentage=0.5,
        #                              CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (50%)"),
        get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'],
                                     irrelevant_selector_threshold=0, irrelevant_selector_percentage=0.4,
                                     CV=Keys.stratified,
                                     held_out_fold=True, pretty_name="Channel Selection (40%)"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=0.5)"),
        ]
    configs_stratified_Fz_reference = [
        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        held_out_fold=True, pretty_name="Baseline (all channels)", Fz_reference=True),

        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        included_channels="no_wearables",
                        held_out_fold=True, pretty_name="Baseline (full-scalp EEG)", Fz_reference=True),

        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.stratified,
                        included_channels="CROSStop",
                        held_out_fold=True, pretty_name="Baseline (CROSStop SD)", Fz_reference=True),
        # get_base_config(base_dir, unique_locations, suffix="rerun", CV=Keys.stratified,
        #                 held_out_fold=True, pretty_name="Baseline"),
        get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'], CV=Keys.stratified,
                                     irrelevant_selector_threshold=0, Fz_reference=True,
                                     held_out_fold=True, pretty_name="Channel Selection"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=0.5)", Fz_reference=True),
    ]
    configs_stratified_final_model_reuse = [
        get_base_config(base_dir, unique_locations, suffix="_final_model_reuse_base", included_channels="all",
                        held_out_fold=True, pretty_name="Baseline (held-out fold)"),
        get_base_config(base_dir, unique_locations, suffix="_final_model_reuse", included_channels="Cross_T7",
                        held_out_fold=True, pretty_name="CROSStop SD and T7 (held-out fold)"),
        ]
    configs_stratified_final_model = [
        get_base_config(base_dir, unique_locations, suffix=suffix_ + "_final_model_all", held_out_fold=True,
                                 included_channels='all', pretty_name="Baseline (held-out fold)"),
        get_base_config(base_dir, unique_locations, suffix=suffix_ + "_final_model_Cross_T7", held_out_fold=True,
                                 included_channels='Cross_T7', pretty_name="CROSStop SD and T7 (held-out fold)"),
        get_base_config(base_dir, unique_locations, suffix=suffix_ + "_final_model_CROSStop", held_out_fold=True,
                                 included_channels='CROSStop', pretty_name="CROSStop SD (held-out fold)"),
    ]
    configs_loho = [
        get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.leave_one_hospital_out,
                        held_out_fold=True, pretty_name="Baseline"),
        # get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'], CV=Keys.leave_one_hospital_out,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=-100)"),
        get_channel_selection_config(base_dir, locations=unique_locations, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'],
                                     irrelevant_selector_threshold=0.5, CV=Keys.leave_one_hospital_out,
                                     held_out_fold=True, pretty_name="Channel Selection"),
        ]
    configs_loho_final_model_reuse = [
        get_base_config(base_dir, unique_locations, suffix="_final_model_reuse_base", included_channels="all",
                        held_out_fold=True, CV=Keys.leave_one_hospital_out, pretty_name="Baseline (held-out fold)"),
        get_base_config(base_dir, unique_locations, suffix="_final_model_reuse", included_channels="Cross_T7",
                        held_out_fold=True, CV=Keys.leave_one_hospital_out,
                        pretty_name="CROSStop SD and T7 (held-out fold)"),
    ]
    configs_ = configs_stratified
    # configs_ = configs_stratified_Fz_reference
    # configs_ = configs_stratified_final_model
    # configs_ = configs_loho + configs_loho_final_model_reuse
    # configs_wearables = [
    #     get_base_config(base_dir, suffix=suffix_, included_channels='wearables', pretty_name="Baseline"),
    #     get_channel_selection_config(base_dir, suffix=suffix_, included_channels='wearables', pretty_name="Channel Selection (AUROC)"),
    #     get_channel_selection_config(base_dir, suffix=suffix_, evaluation_metric=evaluation_metrics['score'],
    #                                  included_channels='wearables', pretty_name="Channel Selection (Score)"), ]

    if 'dtai' in base_dir:
        output_path_base = os.path.join('/cw/dtailocal/loren/2025-Epilepsy', 'figures', 'plots_scores')
    else:
        output_path_base = os.path.join(base_dir, 'figures', 'plots_scores')

    if task == 'violin':
        output_path_ = os.path.join(output_path_base, "violin_plots")
        assert metric_ != 'all', "Metric 'all' is not supported for violin plots. Please choose a specific metric."
        violin_plot(configs_, metric=metric_,
                    output_path=output_path_,
                    threshold=threshold_)
        # violin_plot(configs_wearables, metric=metric_,
        #             output_path=os.path.join(output_path_, f"wearables.png"),
        #             threshold=threshold_)
    elif task == 'thresholds':
        if metric_ == 'all':
            metrics_ = ['f1_ovlp', 'fah_ovlp', 'fah_epoch', 'prec_ovlp', 'sens_ovlp', 'score']
        else:
            metrics_ = [metric_]
        common_name = find_longest_common_substring([config.get_name() for config in configs_])
        output_path_ = os.path.join(output_path_base, f"varying_thresholds_{common_name}")
        plot_varying_thresholds(configs_, metrics=metrics_, output_path=output_path_, rmsa_filtering=not args.no_rmsa,
                                split_localization=True)

    if os.path.exists("net/"):
        shutil.rmtree("net/")