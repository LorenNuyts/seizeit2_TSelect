import argparse
import shutil

from analysis.channel_analysis import *
from analysis.channel_analysis.file_management import download_remote_configs
from net.DL_config import get_channel_selection_config
from utility.constants import evaluation_metrics, Locations, parse_location

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str,
                        choices=["frequent_channels", "count_channels"])
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
                # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_),
                # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
                #                              included_channels='wearables'),
                get_channel_selection_config(base_dir, locations=locations_, suffix=suffix_,
                                             evaluation_metric=evaluation_metrics['score'],
                                             irrelevant_selector_threshold=0.5),
                # get_channel_selection_config(base_dir, locations=args.locations, suffix=suffix_,
                #                              included_channels='wearables',
                #                              evaluation_metric=evaluation_metrics['score'])
    ]

    download_remote_configs(configs_, local_base_dir=configs_[0].save_dir)

    if 'dtai' in base_dir:
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy','analysis', 'results')
    else:
        output_path_ = os.path.join(base_dir, 'analysis', 'results')

    if args.task == "frequent_channels":
        mine_frequent_channels(base_dir, configs_, output_path=output_path_, min_support=0.2)
    elif args.task == "count_channels":
        count_selected_channels_across_folds(base_dir, configs_, output_path=output_path_)
    else:
        raise ValueError(f"Unknown task: {args.task}. Choose from 'frequent_channels' or 'count_channels'.")

    # Clean up the net directory if it exists. It should not be created in the first place, but if it is, we remove it.
    if os.path.exists("net/"):
        shutil.rmtree("net/")