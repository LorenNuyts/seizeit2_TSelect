import argparse
import os
import shutil
from typing import List

from matplotlib import pyplot as plt

from net.DL_config import Config, get_channel_selection_config
from utility.constants import evaluation_metrics, Locations, parse_location
from utility.paths import get_path_results
from utility.stats import Results

base_dir = os.path.dirname(os.path.realpath(__file__))

def count_selected_channels_across_folds(configs: List[Config], output_path: str = None):
    for config in configs:
        results_path = os.path.join(base_dir, "..", get_path_results(config, config.get_name()))
        results = Results(config)
        if os.path.exists(results_path):
            results.load_results(results_path)
        else:
            print(f"Results not found for {config.get_name()}")
            continue
        nb_folds = len(results.config.selected_channels.keys())
        channels = results.config.selected_channels
        channel_count = {}
        for fold in channels:
            for ch in channels[fold]:
                if ch not in channel_count:
                    channel_count[ch] = 1
                else:
                    channel_count[ch] += 1

        sorted_channels = sorted(channel_count.items(), key=lambda x: x[1], reverse=True)
        print(f"Number of folds a channel is chosen: {sorted_channels}")
        # make a hbar plot with channels on y-axis and counts on x-axis + sort them
        channels, counts = zip(*sorted_channels)

        # Create horizontal bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(channels, counts, color='skyblue')
        plt.xlabel("Number of folds the channel is selected out of the {} folds".format(nb_folds))
        plt.title("Channel Counts")
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top

        # if output_path:
        #     output_file = os.path.join(output_path, f"{config.get_name()}_channel_counts.png")
        #     if not os.path.exists(output_path):
        #         os.makedirs(output_path)
        #     plt.savefig(output_file)
        # else:
        #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, nargs="?", default="")
    parser.add_argument(
        '--locations',
        nargs='+',  # accept multiple inputs
        type=parse_location,
        default=[Locations.leuven_adult],
        help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
             f"Defaults to [{Locations.leuven_adult}]."
    )
    args = parser.parse_args()
    suffix_ = args.suffix
    configs_ = [
                # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_),
                # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
                #                              included_channels='wearables'),
                get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
                                             evaluation_metric=evaluation_metrics['score']),
                get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
                                             included_channels='wearables',
                                             evaluation_metric=evaluation_metrics['score'])]

    if 'dtai' in base_dir:
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy', 'figures', 'channel_counts')
    else:
        output_path_ = os.path.join(base_dir, 'figures', 'channel_counts')

    count_selected_channels_across_folds(configs_, output_path=output_path_)
    if os.path.exists("net/"):
        shutil.rmtree("net/")
