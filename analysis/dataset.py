import argparse
import os

import pandas as pd
import pyedflib

from utility.constants import Nodes, Paths, Locations

base_dir = os.path.dirname(os.path.realpath(__file__))

def channel_names(root_dir, locations):
    known_nodes = (Nodes.basic_eeg_nodes + Nodes.optional_eeg_nodes + Nodes.wearable_nodes + Nodes.eeg_ears +
                   Nodes.eeg_acc + Nodes.eeg_gyr + Nodes.ecg_emg_nodes +
                   Nodes.other_nodes + Nodes.ecg_emg_acc + Nodes.ecg_emg_gyr)
    unknown_nodes = dict()
    for location in os.listdir(root_dir):
        if locations is not None and location not in locations:
            continue
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            for recording in os.listdir(subject_path):
                if recording.endswith(".edf"):
                    edf_file = os.path.join(subject_path, recording)
                    with pyedflib.EdfReader(edf_file) as edf:
                        channels_in_file = edf.getSignalLabels()
                        try:
                            channels = Nodes.match_nodes(channels_in_file, known_nodes)
                            for ch in channels:
                                if ch not in known_nodes:
                                    if location in unknown_nodes:
                                        unknown_nodes[location].add(ch)
                                    else:
                                        unknown_nodes[location] = {ch}
                        except (ValueError, AssertionError) as e:
                            print("Error enctountered", e)
                            continue
    print("Unknown nodes found in the dataset:")
    print(unknown_nodes)

def subjects_with_seizures(root_dir, locations):
    seizure_subjects = set()
    for location in os.listdir(root_dir):
        if locations is not None and location not in locations:
            continue
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            for recording in os.listdir(subject_path):
                if subject in seizure_subjects:
                    break
                if recording.endswith(".tsv"):
                    tsv_file = os.path.join(subject_path, recording)
                    df = pd.read_csv(tsv_file, delimiter="\t", skiprows=4)
                    for i, e in df.iterrows():
                        if e['class'] == 'seizure' and e['main type'] == 'focal':
                            seizure_subjects.add(subject)
                            break
    print("Subjects with seizures found in the dataset:")
    print(seizure_subjects)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["channel_names", "subjects_with_seizures"],)
    parser.add_argument("--locations", type=str, nargs="?", default="all")
    args = parser.parse_args()

    if args.locations == 'all':
        locations_ = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]
    else:
        try:
            locations_ = [getattr(Locations, args.locations)]
        except AttributeError:
            raise ValueError(f"Unknown location: {args.locations}")

    if 'dtai' in base_dir:
        data_path = Paths.remote_data_path
    else:
        data_path = Paths.local_data_path

    if args.task == "subjects_with_seizures":
        subjects_with_seizures(data_path, locations=locations_)
    elif args.task == "channel_names":
        channel_names(data_path, locations=locations_)
    else:
        raise ValueError(f"Unknown task: {args.task}. Use 'channel_names' or 'subjects_with_seizures'.")
