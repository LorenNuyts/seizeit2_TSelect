import argparse
import os
from typing import List

import pandas as pd

from utility.constants import evaluation_metrics, parse_location, Locations, Keys
from net.DL_config import get_base_config, get_channel_selection_config
from utility.paths import get_path_results
from utility.stats import Results

base_dir = os.path.dirname(os.path.realpath(__file__))

def print_table(configs: list, metrics: List[str], output_path: str):
    data = {}

    for config in configs:
        results_path = os.path.join(base_dir, "..", get_path_results(config, config.get_name()))
        results = Results(config)
        print("Now handling: ", results_path)
        if os.path.exists(results_path):
            results.load_results(results_path)
        else:
            print(f"Results not found for {config.get_name()}")
            continue

        values = []
        for metric in metrics:
            value = getattr(results, metric, None)
            if value is None:
                formatted = "N/A"
            else:
                # Derive std metric name
                if "median" in metric:
                    std_metric = metric.replace("median", "std")
                elif "average" in metric:
                    std_metric = metric.replace("average", "std")
                else:
                    std_metric = None

                if std_metric:
                    std = getattr(results, std_metric, None)
                else:
                    std = None

                if std is not None:
                    formatted = f"{value:.3f} ± {std:.2f}"
                else:
                    formatted = f"{value:.3f}"

            values.append(formatted)

        data[config.get_name()] = values

    df = pd.DataFrame(data, index=metrics)
    print(df.to_csv(sep='\t', index=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    parser.add_argument(
        '--locations',
        nargs='+',  # accept multiple inputs
        type=parse_location,
        default=[parse_location(l) for l in Locations.all_keys()],
        help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
             f"Defaults to [{Locations.all_keys()}]."
    )
    args = parser.parse_args()

    locations_ = sorted(list(dict.fromkeys(args.locations)))
    suffix_ = args.suffix
    configs_ = [
        # get_base_config(base_dir, locations=locations_, suffix=suffix_, CV=Keys.stratified,
        #                 held_out_fold=True, pretty_name="Base"),
        # get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'], CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=-100)"),
        # get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0.5, CV=Keys.stratified,
        #                              held_out_fold=True, pretty_name="Channel Selection (th=0.5)"),
        # get_base_config(base_dir, unique_locations, suffix=suffix_, CV=Keys.leave_one_hospital_out,
        #                 held_out_fold=True, pretty_name="Baseline"),
        get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'], CV=Keys.leave_one_hospital_out,
                                     held_out_fold=True, pretty_name="Channel Selection (th=-100)"),
        get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
                                     evaluation_metric=evaluation_metrics['score'],
                                     irrelevant_selector_threshold=0.5, CV=Keys.leave_one_hospital_out,
                                     held_out_fold=True, pretty_name="Channel Selection (th=0.5)"),
        # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_),
        # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
        #                              included_channels='wearables'),
        # get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0.5),
        # get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
        #                              evaluation_metric=evaluation_metrics['score'],
        #                              irrelevant_selector_threshold=0.5, CV=Keys.leave_one_hospital_out),
        # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
        #                              included_channels='wearables',
        #                              evaluation_metric=evaluation_metrics['score'])
    ]
    # configs_base = [
    #     get_base_config(base_dir, suffix=suffix_),
    #             get_channel_selection_config(base_dir, suffix=suffix_),
    #             get_channel_selection_config(base_dir, suffix=suffix_, evaluation_metric=evaluation_metrics['score']),]
    # configs_wearables = [
    #     get_base_config(base_dir, suffix=suffix_, included_channels='wearables'),
    #             get_channel_selection_config(base_dir, suffix=suffix_, included_channels='wearables'),
    #             get_channel_selection_config(base_dir, suffix=suffix_, evaluation_metric=evaluation_metrics['score'],
    #                                             included_channels='wearables'),]
    metrics_ = ['average_nb_channels', 'average_selection_time', 'average_train_time', 'average_total_time',
        # 'average_f1_ovlp_best_threshold', 'average_fah_ovlp_best_threshold', 'average_fah_epoch_best_threshold',
        #         'average_prec_ovlp_best_threshold',
        #         'average_sens_ovlp_best_threshold', 'average_rocauc_best_threshold', 'average_score_best_threshold',

                # 'average_f1_ovlp_th05', 'average_fah_ovlp_th05', 'average_fah_epoch_th05',
                # 'average_prec_ovlp_th05', 'average_sens_ovlp_th05', 'average_rocauc_th05', 'average_score_th05',
                # 'median_f1_ovlp_best_threshold', 'median_fah_ovlp_best_threshold', 'median_fah_epoch_best_threshold',
                # 'median_prec_ovlp_best_threshold',
                # 'median_sens_ovlp_best_threshold', 'median_rocauc_best_threshold', 'median_score_best_threshold',
                # 'median_f1_ovlp_th05', 'median_fah_ovlp_th05', 'median_fah_epoch_th05',
                # 'median_prec_ovlp_th05', 'median_sens_ovlp_th05',
                # 'median_rocauc_th05', 'median_score_th05'
                ]
    # metrics_ = ['average_nb_channels', 'average_total_time',
    #             'median_score_th05', 'median_sens_ovlp_th05', 'median_fah_epoch_th05', 'median_fah_ovlp_th05',
    #             'median_score_best_threshold',
    #             'median_sens_ovlp_best_threshold', 'median_fah_epoch_best_threshold', 'median_fah_ovlp_best_threshold',
    #             ]

    if 'dtai' in base_dir:
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy', 'tables', 'results.xlsx')
    else:
        output_path_ = os.path.join(base_dir, 'tables', 'results.xlsx')

    if not os.path.exists(os.path.dirname(output_path_)):
        os.makedirs(os.path.dirname(output_path_))

    print("Base table: ")
    print_table(configs_, metrics_, output_path_)
    # print("########################################################\n")
    # print("Wearables table: ")
    # print_table(configs_wearables, metrics_, output_path_)



