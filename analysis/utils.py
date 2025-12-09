import os
from collections import defaultdict
from typing import List, Any

from analysis.channel_analysis import download_remote_configs, download_remote_results
from net.DL_config import Config
from utility.paths import get_path_results
from utility.stats import Results


def find_longest_common_substring(strs: List[str]):
    if not strs:
        return ""

    shortest_str = min(strs, key=len)
    length = len(shortest_str)

    for sub_len in range(length, 0, -1):
        for start in range(length - sub_len + 1):
            substring = shortest_str[start:start + sub_len]
            if all(substring in s for s in strs):
                return substring
    return ""


def extract_values_std_from_results(base_dir: str, configs: list[Config], full_to_short_names: dict[Any, Any], lat: str,
                                    metrics: list[str], rmsa_filtering: bool) -> tuple[
    defaultdict[str, dict], list[float], int, int]:
    data = defaultdict(dict)
    thresholds = None
    for metric in metrics:
        for config in configs:
            results_path = os.path.join(base_dir, "..",
                                        get_path_results(config, config.get_name() + (f"_{lat}" if lat else "")))
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
    lower_bound = len(
        thresholds) - 1  # Last threshold because the lower bound should be the lowest one of all configurations
    upper_bound = 0  # First threshold because the upper bound should be the highest one of all configurations
    for label, values in data["fah_epoch"].items():
        max_fah = max(values["average"])
        lower_bound_i = len(thresholds) - 1 - values["average"][::-1].index(max_fah)
        lower_bound = min(lower_bound, lower_bound_i)
        upper_bound_i = values["average"].index(0.0)
        if upper_bound_i is None:
            upper_bound_i = len(thresholds) - 1
        upper_bound = max(upper_bound, upper_bound_i)

    thresholds = thresholds[lower_bound:upper_bound]
    return data, thresholds, lower_bound, upper_bound


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


pretty_print_metrics = {'f1_ovlp': 'F1 (overlap)', 'fah_ovlp': 'False alarm rate (overlap)',
                        'fah_epoch': 'False alarm rate (epoch)',
                        'prec_ovlp': 'Precision (overlap)', 'sens_ovlp': 'Sensitivity (overlap)', 'score': 'Score',}
