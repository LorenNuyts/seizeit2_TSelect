import argparse
import os

from tqdm import tqdm
from data.data import Data
from net.DL_config import get_base_config
from net.key_generator import generate_data_keys_sequential
from utility import get_recs_list

from utility.constants import parse_location, Locations
from utility.dataset_management import create_single_tfrecord
from utility.paths import get_path_preprocessed_data

base_ = os.path.dirname(os.path.realpath(__file__))

def create_preprocessed_dataset(root_dir, config):
    """
    Creates a preprocessed dataset and saves it in the 'Preprocessed' folder.

    Args:
        root_dir (str): Root directory containing the dataset.
        config (cls): Configuration object with experiment parameters.
    """
    for location in config.locations:
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

                preprocessed_file = get_path_preprocessed_data(root_dir, recording)
                if os.path.exists(preprocessed_file):
                    print(f"         | Preprocessed data already exists at {preprocessed_file}, skipping.")
                    continue

                rec_data = Data.loadData(root_dir, recording,
                                         included_channels=config.included_channels)
                rec_data.apply_preprocess(config)

                rec_data.store_h5(preprocessed_file)
                print(f"         | Saved preprocessed data to {preprocessed_file}")


def create_tfrecord_dataset(root_dir, config):
    """
    Creates preprocessed TFRecord dataset and saves segments in TFRecord format.

    Args:
        root_dir (str): Directory containing the raw data recordings.
        config (cls): Configuration object with preprocessing + segmentation settings.
    """
    if Locations.leuven_adult in config.locations:
        included_subjects = ['SUBJ-1a-159', 'SUBJ-1a-358', 'SUBJ-1a-153']  # Leuven Adult subjects
    else:
        included_subjects = ['SUBJ-7-379', 'SUBJ-7-376']  # Coimbra subjects
    for location in config.locations:
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        if not os.path.isdir(location_path):
            continue

        for subject in os.listdir(location_path):
            if subject not in included_subjects:
                continue
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            if not os.path.isdir(subject_path):
                continue

            recs = get_recs_list(root_dir, [location], [subject])
            segments = generate_data_keys_sequential(config, recs)


            for s in tqdm(segments, desc="Writing TFRecord"):
                create_single_tfrecord(config, recs, s)

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
    # create_preprocessed_dataset(config_.data_path, config_)
    create_tfrecord_dataset(config_.data_path, config_)