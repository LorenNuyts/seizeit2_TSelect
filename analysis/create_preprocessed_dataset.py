import argparse
import os

import h5py
import pyedflib
from data.data import Data
from net.DL_config import Config, get_base_config
from utility import get_recs_list

from utility.constants import parse_location, Locations
from utility.paths import get_path_recording, get_path_preprocessed_data

base_ = os.path.dirname(os.path.realpath(__file__))

def create_preprocessed_dataset(root_dir, config):
    """
    Creates a preprocessed dataset and saves it in the 'Preprocessed' folder.

    Args:
        root_dir (str): Root directory containing the dataset.
        config (cls): Configuration object with experiment parameters.
    """
    for location in os.listdir(root_dir):
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        if not os.path.isdir(location_path):
            continue

        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            if not os.path.isdir(subject_path):
                continue
            recs = get_recs_list(root_dir, [location], [subject])
            for recording in recs:
                print("         | Processing recording:", recording)
                rec_data = Data.loadData(root_dir, recording,
                                         included_channels=config.included_channels)
                rec_data.apply_preprocess(config)

                preprocessed_file = get_path_preprocessed_data(root_dir, recording)
                rec_data.store_h5(preprocessed_file)
                print(f"         | Saved preprocessed data to {preprocessed_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locations',
        nargs='+',  # accept multiple inputs
        type=parse_location,
        default=[Locations.leuven_adult],
        help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
             f"Defaults to [{Locations.leuven_adult}]."
    )
    args = parser.parse_args()

    unique_locations = list(dict.fromkeys(args.locations))
    # Import your configuration class
    config_ = get_base_config(base_, locations=unique_locations,)
    create_preprocessed_dataset(config_.data_path, config_)