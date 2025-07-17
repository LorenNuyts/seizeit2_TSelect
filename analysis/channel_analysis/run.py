import argparse
import shutil

from analysis.channel_analysis import *
from analysis.channel_analysis.file_management import download_remote_configs
from net.DL_config import get_channel_selection_config
from utility.constants import evaluation_metrics, Locations, parse_location

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
        output_path_ = os.path.join('/cw/dtailocal/loren/2025-Epilepsy','analysis', 'figures')
    else:
        output_path_ = os.path.join(base_dir, 'analysis', 'figures')

    count_selected_channels_across_folds(base_dir, configs_, output_path=output_path_)
    if os.path.exists("net/"):
        shutil.rmtree("net/")