import argparse
import os
from typing import List

import pandas as pd
import pyedflib

from data.data import switch_channels
from utility.constants import Nodes, Paths, Locations

base_dir = os.path.dirname(os.path.realpath(__file__))

def channel_names(root_dir, locations):
    # known_nodes = (Nodes.basic_eeg_nodes + Nodes.optional_eeg_nodes + Nodes.wearable_nodes + Nodes.eeg_ears +
    #                Nodes.eeg_acc + Nodes.eeg_gyr + Nodes.ecg_emg_nodes +
    #                Nodes.other_nodes + Nodes.ecg_emg_sd_acc + Nodes.ecg_emg_sd_gyr)
    known_nodes = Nodes.all_nodes()
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
                                    print(f"Unknown channel found: {ch}")
                                    if location in unknown_nodes:
                                        unknown_nodes[location].add(ch)
                                    else:
                                        unknown_nodes[location] = {ch}
                        except (ValueError, AssertionError) as e:
                            print("Error encountered", e)
                            continue
    print("Unknown nodes found in the dataset:")
    print(unknown_nodes)

def subjects_with_seizures(root_dir, locations):
    seizure_subjects = {loc: set() for loc in locations}
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
                            seizure_subjects[location].add(subject)
                            # seizure_subjects.add(subject)
                            break
    print("Subjects with seizures found in the dataset:")
    print(seizure_subjects)
    print({loc: len(seizure_subjects[loc]) for loc in seizure_subjects})

def seizure_segments_per_location(root_dir, locations):
    seizure_segments = {loc: 0 for loc in locations}
    for location in os.listdir(root_dir):
        if locations is not None and location not in locations:
            continue
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            for recording in os.listdir(subject_path):
                if recording.endswith(".tsv"):
                    tsv_file = os.path.join(subject_path, recording)
                    df = pd.read_csv(tsv_file, delimiter="\t", skiprows=4)
                    for i, e in df.iterrows():
                        if e['class'] == 'seizure' and e['main type'] == 'focal':
                            seizure_segments[location] += int(2*(e['stop'] - e['start']))
    print("Seizure segments per location:")
    print(seizure_segments)

def channel_differences_between_subjects(root_dir, locations: List[str]):
    ref_channels = set(Nodes.basic_eeg_nodes)
    channel_differences = dict()
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
                        formatted_channels = set(Nodes.match_nodes(channels_in_file, list(ref_channels)))
                        switched_channels = set(switch_channels(list(ref_channels), list(formatted_channels),  Nodes.switchable_nodes))
                        if not ref_channels.issubset(switched_channels):
                            if location not in channel_differences:
                                channel_differences[location] = dict()
                            channel_differences[location][f"{subject}_{recording}"] = ref_channels - set(switched_channels)
                            print(f"Channel differences: {ref_channels - set(switched_channels)}")
    print("Reference channels per location:")
    print(ref_channels)
    print("Channel differences between subjects:")
    print(channel_differences)


def rank_locations(root_dir, locations: List[str]):
    location_counts = {loc: [0,0] for loc in locations}
    location_average_time_steps = {}
    for location in os.listdir(root_dir):
        time_steps = 0
        nb_recordings = 0
        if locations is not None and location not in locations:
            continue
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        location_counts[location][1] = len(os.listdir(location_path))
        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            for recording in os.listdir(subject_path):
                if recording.endswith(".tsv"):
                    df = pd.read_csv(os.path.join(subject_path, recording), delimiter="\t", skiprows=4)
                    for i, e in df.iterrows():
                        if e['class'] == 'seizure' and e['main type'] == 'focal':
                            location_counts[location][0] += 1
                elif recording.endswith(".edf"):
                    edf_file = os.path.join(subject_path, recording)
                    with pyedflib.EdfReader(edf_file) as edf:
                        n_samples = edf.getNSamples()[0]
                        time_steps += n_samples
                        nb_recordings += 1
        if nb_recordings > 0:
            average_time_steps = time_steps / nb_recordings
            location_average_time_steps[location] = average_time_steps
            print(f"Average time steps for {location}: {average_time_steps}")

        # Sort locations by number of seizures and number of subjects, from highest to lowest
        sorted_locations = sorted(location_counts.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
        print("Locations ranked by number of seizures and subjects:")
        for loc, counts in sorted_locations:
            print(f"{loc}: {counts[0]} seizures, {counts[1]} subjects")

def average_memory_size_subject(root_dir, locations, channels=Nodes.basic_eeg_nodes):
    total_memory = 0
    total_subjects = 0
    n_channels = len(channels)
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
                        n_samples = edf.getNSamples()[0]
                        memory_size = n_channels * n_samples * 8  # Assuming 8 bytes per sample (float64)
                        total_memory += memory_size
                        total_subjects += 1
    average_memory = total_memory / total_subjects if total_subjects > 0 else 0
    print(f"Average memory size per subject: {average_memory / (1024 * 1024):.2f} MB")

def hours_of_data_per_location(root_dir, locations):
    total_hours = {loc: 0 for loc in locations}
    for location in locations:
        print("Processing location:", location)
        location_path = os.path.join(root_dir, location)
        for subject in os.listdir(location_path):
            print("     | Processing subject:", subject)
            subject_path = os.path.join(location_path, subject)
            for recording in os.listdir(subject_path):
                if recording.endswith(".edf"):
                    edf_file = os.path.join(subject_path, recording)
                    with pyedflib.EdfReader(edf_file) as edf:
                        n_samples = edf.getNSamples()[0]
                        sampling_frequency = edf.getSampleFrequency(0)  # Assuming all channels have the same frequency
                        duration_seconds = n_samples / sampling_frequency
                        total_hours[location] += duration_seconds / 3600  # Convert seconds to hours
    print("Total hours of data per location:")
    print(total_hours)

def dataset_stats(data_path: str, save_dir: str, locations: List[str] = None):
    if locations is None:
        locations = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = []
    missing_locations = []
    for location in locations:
        file_name = os.path.join(save_dir, "dataset_stats_" + location + ".csv")
        if os.path.exists(file_name):
            data.append(pd.read_csv(file_name))
        else:
            missing_locations.append(location)

    for location in missing_locations:
        file_name = os.path.join(save_dir, "dataset_stats_" + location + ".csv")
        info_per_subject = {'subject': [],
                            'hospital': [],
                            'n_seizures': [],
                            'hours_of_data': []}
        print(f"Processing location: {location}")
        location_path = os.path.join(data_path, location)
        if not os.path.isdir(location_path):
            print(f"Location {location} does not exist in the dataset.")
            continue
        subjects = os.listdir(location_path)
        for subject in subjects:
            n_seizures = 0
            hours_of_data = 0
            subject_path = os.path.join(location_path, subject)
            recordings = [f for f in os.listdir(subject_path) if f.endswith('.edf')]
            for recording in recordings:
                edf_file = os.path.join(subject_path, recording)
                with pyedflib.EdfReader(edf_file) as edf:
                    n_samples = edf.getNSamples()[0]
                    sampling_frequency = edf.getSampleFrequency(0)
                    duration_seconds = n_samples / sampling_frequency
                    hours_of_data += duration_seconds / 3600  # Convert seconds to hours
            tsv_files = [f for f in os.listdir(subject_path) if f.endswith('.tsv')]
            for tsv_file in tsv_files:
                tsv_path = os.path.join(subject_path, tsv_file)
                df = pd.read_csv(tsv_path, delimiter="\t", skiprows=4)
                n_seizures += df[df['class'] == 'seizure'].shape[0]
            info_per_subject['subject'].append(subject)
            info_per_subject['hospital'].append(location)
            info_per_subject['n_seizures'].append(n_seizures)
            info_per_subject['hours_of_data'].append(hours_of_data)

        df = pd.DataFrame(info_per_subject)
        df.to_csv(file_name, index=False)
        data.append(df)

    return pd.concat(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["channel_names", "subjects_with_seizures",
                                                   "channel_differences", "rank_locations", "average_memory_size",
                                                   "seizure_segments", "hours_of_data", "compute_dataset_stats"],)
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
        data_path_ = Paths.remote_data_path
        save_dir_ = Paths.remote_save_dir
    else:
        data_path_ = Paths.local_data_path
        save_dir_ = Paths.local_save_dir

    if args.task == "subjects_with_seizures":
        subjects_with_seizures(data_path_, locations=locations_)
    elif args.task == "channel_names":
        channel_names(data_path_, locations=locations_)
    elif args.task == "channel_differences":
        channel_differences_between_subjects(data_path_, locations=locations_)
    elif args.task == "rank_locations":
        rank_locations(data_path_, locations=locations_)
    elif args.task == "average_memory_size":
        average_memory_size_subject(data_path_, locations=locations_, channels=Nodes.basic_eeg_nodes)
    elif args.task == "seizure_segments":
        seizure_segments_per_location(data_path_, locations=locations_)
    elif args.task == "hours_of_data":
        hours_of_data_per_location(data_path_, locations=locations_)
    elif args.task == "compute_dataset_stats":
        dataset_stats(data_path_, os.path.join(base_dir, "..", save_dir_, "dataset_stats"), locations=locations_)
    else:
        raise ValueError(f"Unknown task: {args.task}. Use 'channel_names' or 'subjects_with_seizures'.")
