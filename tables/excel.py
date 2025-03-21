import argparse
import os
from typing import List

import pandas as pd

from utility.constants import get_base_config, get_channel_selection_config
from utility.paths import get_path_results
from utility.stats import Results

base_dir = os.path.dirname(os.path.realpath(__file__))

def print_table(configs: list, metrics: List[str], output_path: str):
    data = {}

    for config in configs:
        results_path = os.path.join(base_dir, "..", get_path_results(config, config.get_name()))
        results = Results(config)
        if os.path.exists(results_path):
            results.load_results(results_path)
        else:
            print(f"Results not found for {config.get_name()}")
            continue
        data[config.get_name()] = [getattr(results, metric, None) for metric in metrics]

    df = pd.DataFrame(data, index=metrics)

    print(df.to_csv(sep='\t', index=True))
    df.to_excel(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    args = parser.parse_args()
    suffix_ = args.suffix
    configs_ = [get_base_config(base_dir, suffix=suffix_), get_channel_selection_config(base_dir, suffix=suffix_)]
    metrics_ = ['average_f1_ovlp_best_threshold', 'average_fah_ovlp_best_threshold', 'average_prec_ovlp_best_threshold',
                'average_sens_ovlp_best_threshold', 'average_score_best_threshold',
                'best_average_f1_ovlp', 'best_average_fah_ovlp', 'best_average_prec_ovlp', 'best_average_sens_ovlp',
                'best_average_score', 'average_selection_time', 'average_train_time', 'average_total_time', ]
    if 'dtai' in base_dir:
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy', 'tables', 'results.xlsx')
    else:
        output_path_ = os.path.join(base_dir, 'tables', 'results.xlsx')

    if not os.path.exists(os.path.dirname(output_path_)):
        os.makedirs(os.path.dirname(output_path_))

    print_table(configs_, metrics_, output_path_)


