import argparse
import os

import numpy as np
import pyedflib
from tqdm import tqdm
from data.data import Data, create_single_tfrecord, switch_channels
from net.DL_config import get_base_config
from net.key_generator import generate_data_keys_sequential
from net.utils import rereference_average_signal
from utility import get_recs_list

from utility.constants import parse_location, Locations, Nodes
from utility.paths import get_path_preprocessed_data, get_path_recording

base_ = os.path.dirname(os.path.realpath(__file__))

def create_preprocessed_dataset(root_dir, config):
    """
    Creates a preprocessed dataset and saves it in the 'Preprocessed' folder.

    Args:
        root_dir (str): Root directory containing the dataset.
        config (cls): Configuration object with experiment parameters.
    """
    for location in config.locations:
    # for location in [Locations.coimbra]:
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        if not os.path.isdir(location_path):
            continue

        subj = ['SUBJ-7-329']
        for subject in os.listdir(location_path):
        # for subject in subj:
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
                rec_data.apply_preprocess(data_path=config.data_path, fs=config.fs)

                rec_data.store_h5(preprocessed_file)
                print(f"         | Saved preprocessed data to {preprocessed_file}")


def create_rereferenced_dataset(root_dir, config):
    """
    Creates a preprocessed dataset and saves it in the 'Preprocessed' folder.

    Args:
        root_dir (str): Root directory containing the dataset.
        config (cls): Configuration object with experiment parameters.
    """

    def safe_physical_limit(x, digits=6):
        """Round to at most 6 significant digits to fit in EDF+ 8-char limit."""
        return float(f"{x:.6g}")

    for location in config.locations:
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        if not os.path.isdir(location_path):
            continue

        # subj = ['SUBJ-1b-315']
        nb_subjects = 3
        subj = np.random.choice(os.listdir(location_path), nb_subjects, replace=False)
        # for subject in os.listdir(location_path):
        for subject in subj:
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            if not os.path.isdir(subject_path):
                continue
            recs = get_recs_list(root_dir, [location], [subject])
            if len(recs) > 2:
                recs_i = np.random.choice(range(len(recs)), 2, replace=False)
            else:
                recs_i = range(len(recs))
            for rec_i in recs_i:
                recording = recs[rec_i]
                print("         | Processing recording:", recording)


                preprocessed_file = get_path_preprocessed_data(root_dir, recording)
                rereferenced_path = os.path.join(root_dir, 'Rereferenced', recording[0], recording[1], f"{recording[1]}_{recording[2]}_rereferenced.edf")
                os.makedirs(os.path.dirname(rereferenced_path), exist_ok=True)
                # if os.path.exists(rereferenced_path):
                #     print(f"         | Preprocessed data already exists at {preprocessed_file}, skipping.")
                #     continue

                rec_data = Data.loadData(root_dir, recording,
                                         included_channels=config.included_channels, load_preprocessed=False)
                rereferenced_data = rereference_average_signal(rec_data.data, rec_data.channels)

                edfFile = get_path_recording(root_dir, recording)
                # rereferenced_path = preprocessed_file.split('.h5')[0] + "_rereferenced.edf"
                with pyedflib.EdfReader(edfFile) as original_edf:
                    channels_in_file = original_edf.getSignalLabels()
                    included_channels = rec_data.channels
                    standardized_channels_in_file = switch_channels(channels_in_file, included_channels,
                                                                    Nodes.switchable_channels)
                    # standardized_included_channels = Nodes.match_nodes(included_channels, Nodes.all_nodes())

                    # Build the channel info list (can reuse original parameters)
                    channel_headers = []
                    for i in range(len(rec_data.channels)):
                        original_i = standardized_channels_in_file.index(rec_data.channels[i])
                        original_header = original_edf.getSignalHeader(original_i)
                        channel_headers.append(original_header)

                    general_header = original_edf.getHeader()  # overall EDF header info

                writer = pyedflib.EdfWriter(
                    rereferenced_path,
                    n_channels=len(rec_data.channels),
                    file_type=pyedflib.FILETYPE_EDFPLUS
                )
                writer.setSignalHeaders(channel_headers)
                writer.setHeader(general_header)
                writer.writeSamples(rereferenced_data)
                writer.close()

                print(f"         | Saved preprocessed data to {rereferenced_path}")


def create_tfrecord_dataset(root_dir, config):
    """
    Creates preprocessed TFRecord dataset and saves segments in TFRecord format.

    Args:
        root_dir (str): Directory containing the raw data recordings.
        config (cls): Configuration object with preprocessing + segmentation settings.
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
            segments = generate_data_keys_sequential(config, recs)


            for s in tqdm(segments, desc="Writing TFRecord"):
                create_single_tfrecord(config, recs, s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--locations',
        nargs='+',  # accept multiple inputs
        type=parse_location,
        default=[parse_location(l) for l in Locations.all_keys()],
        # default=[Locations.leuven_adult],
        help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
             f"Defaults to all locations."
    )
    args = parser.parse_args()

    unique_locations = list(dict.fromkeys(args.locations))
    # Import your configuration class
    config_ = get_base_config(base_, locations=unique_locations,)
    create_preprocessed_dataset(config_.data_path, config_)
    # create_rereferenced_dataset(config_.data_path, config_)
    # create_tfrecord_dataset(config_.data_path, config_)