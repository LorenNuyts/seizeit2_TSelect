import argparse
import os

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

def channel_differences_between_subjects(root_dir, locations):
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


def rank_locations(root_dir, locations):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["channel_names", "subjects_with_seizures",
                                                   "channel_differences", "rank_locations", "average_memory_size",
                                                   "seizure_segments"],)
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
    elif args.task == "channel_differences":
        channel_differences_between_subjects(data_path, locations=locations_)
    elif args.task == "rank_locations":
        rank_locations(data_path, locations=locations_)
    elif args.task == "average_memory_size":
        average_memory_size_subject(data_path, locations=locations_, channels=Nodes.basic_eeg_nodes)
    elif args.task == "seizure_segments":
        seizure_segments_per_location(data_path, locations=locations_)
    else:
        raise ValueError(f"Unknown task: {args.task}. Use 'channel_names' or 'subjects_with_seizures'.")
