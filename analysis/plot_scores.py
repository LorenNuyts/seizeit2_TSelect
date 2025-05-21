import argparse
import os
import shutil

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utility.constants import get_base_config, get_channel_selection_config, evaluation_metrics
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
                data.append({
                    "Configuration": full_to_short_names[config.get_name()],
                    "Value": val
                })

            if not data:
                print("No data collected for plotting.")
                return

            df = pd.DataFrame(data)

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x="Configuration", y="Value", inner="box")
            plt.xlabel("")
            plt.ylabel(metric.replace("_", " ").title())
            plt.tight_layout()

            if output_path:
                plot_path = os.path.splitext(output_path)[0] + (f"_{metric.replace('.', '_')}"
                                                                f"_th{str(threshold).replace('.', '')}.png")
                if not os.path.exists(os.path.dirname(plot_path)):
                    os.makedirs(os.path.dirname(plot_path))
                plt.savefig(plot_path)
                print(f"Violin plot saved to {plot_path}")

            plt.show()


if __name__ == '__main__':
    allowed_metrics = {'f1_ovlp', 'fah_ovlp', 'fah_epoch', 'prec_ovlp', 'sens_ovlp', 'rocauc', 'score',
                       'nb_channels', 'selection_time', 'train_time', 'total_time'}
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", type=str,)
    parser.add_argument("--threshold", type=float, default=0.5, nargs="?")
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    args = parser.parse_args()
    metric_ = args.metric
    threshold_ = args.threshold
    suffix_ = args.suffix

    if metric_ not in allowed_metrics:
        raise ValueError(f"Metric {metric_} is not allowed. Choose from {allowed_metrics}")

    configs_base = [
        get_base_config(base_dir, suffix=suffix_),
        get_channel_selection_config(base_dir, suffix=suffix_),
        get_channel_selection_config(base_dir, suffix=suffix_, evaluation_metric=evaluation_metrics['score']), ]
    configs_wearables = [
        get_base_config(base_dir, suffix=suffix_, included_channels='wearables'),
        get_channel_selection_config(base_dir, suffix=suffix_, included_channels='wearables'),
        get_channel_selection_config(base_dir, suffix=suffix_, evaluation_metric=evaluation_metrics['score'],
                                     included_channels='wearables'), ]

    if 'dtai' in base_dir:
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy', 'figures', 'plots_scores')
    else:
        output_path_ = os.path.join(base_dir, 'figures', 'plots_scores')

    violin_plot(configs_base, metric=metric_,
                output_path=os.path.join(output_path_, f"base.png"),
                threshold=threshold_)
    violin_plot(configs_wearables, metric=metric_,
                output_path=os.path.join(output_path_, f"wearables.png"),
                threshold=threshold_)

    if os.path.exists("net/"):
        shutil.rmtree("net/")