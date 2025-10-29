import argparse
import os
import warnings
from collections import defaultdict
from typing import List

import numpy as np
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
                        switched_channels = set(switch_channels(list(ref_channels), list(formatted_channels), Nodes.switchable_channels))
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

def find_subjects_without_channels(data_path: str, channels:List[str], locations: List[str] = None):
    if locations is None:
        locations = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]
    locations = [Locations.freiburg]

    subjects_without_channels = {loc: dict() for loc in locations}
    for location in locations:
        print(f"Processing location: {location}")
        location_path = os.path.join(data_path, location)
        if not os.path.isdir(location_path):
            print(f"Location {location} does not exist in the dataset.")
            continue
        # subjects = os.listdir(location_path)
        subjects = ['SUBJ-4-097']
        for subject in subjects:
            subject_path = os.path.join(location_path, subject)
            recordings = [f for f in os.listdir(subject_path) if f.endswith('.edf')]
            for recording in recordings:
                if subject in subjects_without_channels[location] and len(subjects_without_channels[location][subject]) == len(channels):
                    break
                edf_file = os.path.join(subject_path, recording)
                with pyedflib.EdfReader(edf_file) as edf:
                    channels_subject = edf.getSignalLabels()
                    for ch in channels:
                        if ch not in channels_subject:
                            if subject not in subjects_without_channels[location]:
                                subjects_without_channels[location][subject] = set()
                            subjects_without_channels[location][subject].add(ch)
                            print(f"Subject {subject} in location {location} is missing channel {ch}")

    print("Subjects without required channels:")
    print(subjects_without_channels)
    return subjects_without_channels


def find_ref_channel(data_path, locations: List[str] = None):
    threshold = 0.5
    if locations is None:
        locations = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]
    locations = [Locations.leuven_adult]

    ref_channels = defaultdict(set)
    for location in locations:
        print(f"Processing location: {location}")
        location_path = os.path.join(data_path, location)
        if not os.path.isdir(location_path):
            print(f"Location {location} does not exist in the dataset.")
            continue
        subjects = os.listdir(location_path)
        subjects_done = {'SUBJ-1a-006', "SUBJ-1a-006",  "SUBJ-1a-066", "SUBJ-1a-027", "SUBJ-1a-036", "SUBJ-1a-048",
                        "SUBJ-1a-014",  "SUBJ-1a-068", "SUBJ-1a-015",  "SUBJ-1a-076", "SUBJ-1a-080", "SUBJ-1a-055",
                         "SUBJ-1a-082", "SUBJ-1a-039",  "SUBJ-1a-087", "SUBJ-1a-115",
                        "SUBJ-1a-044"}
        # subjects = ['SUBJ-4-097']
        for subject in subjects:
            if subject in subjects_done:
                continue
            print("     | Processing subject:", subject)
            found_reference = None
            subject_path = os.path.join(location_path, subject)
            recordings = [f for f in os.listdir(subject_path) if f.endswith('.edf')]
            for recording in recordings:
                print(f"        | Processing recording: {recording}")
                edf_file = os.path.join(subject_path, recording)
                with pyedflib.EdfReader(edf_file) as edf:
                    channels = edf.getSignalLabels()
                    for ix, ch in enumerate(channels):
                        if "ACC" in ch or "GYR" in ch or "ECG" in ch or "EMG" in ch:
                            continue
                        data = edf.readSignal(ix)
                        if np.count_nonzero(data == 0) / len(data) >= threshold:
                            if found_reference is not None and found_reference != ch:
                                print(f"Warning: Multiple potential reference channels found for subject {subject} in location {location}: {found_reference} and {ch}")
                            print(f"            Found a reference channel {ch} for subject {subject} in location {location}")
                            found_reference = ch
                            ref_channels[ch].add(subject)

    print("Reference channels:")
    print(ref_channels)
    return ref_channels


def dataset_content(data_path: str, locations: List[str] = None):
    columns = ['Subject ID', 'Hospital', 'Duration Recordings (hours)', 'SD Configuration', 'Affected Lobe', 'FA',
                                  'FIA',
                'FBTC', 'Focal', 'Subclinical', 'Unknown'
               ]
    table = []
    seizure_durations_adult = []
    seizure_durations_pediatric = []
    lateralizations = []

    if locations is None:
        locations = [Locations.coimbra, Locations.freiburg, Locations.aachen, Locations.karolinska,
                     Locations.leuven_adult, Locations.leuven_pediatric]
    # locations = [Locations.coimbra]

    for location in locations:
        print(f"Processing location: {location}")
        location_path = os.path.join(data_path, location)
        if not os.path.isdir(location_path):
            print(f"Location {location} does not exist in the dataset.")
            continue
        subjects = os.listdir(location_path)
        for subject in subjects:
            print("     | Processing subject:", subject)
            table_subject = {'Subject ID': subject, 'Hospital': Locations.to_acronym(location), 'Duration Recordings (hours)': 0,
                             'SD Configuration': None, 'Affected Lobe': None, 'FA': 0, 'FIA': 0,
                              'FBTC': 0, 'Focal': 0, 'Subclinical': 0, 'Unknown': 0
                             }
            subject_path = os.path.join(location_path, subject)
            recordings = [f for f in os.listdir(subject_path) if f.endswith('.edf')]
            for recording in recordings:
                # print(f"        | Processing recording: {recording}")
                edf_file = os.path.join(subject_path, recording)
                with pyedflib.EdfReader(edf_file) as edf:
                    table_subject['Duration Recordings (hours)'] += edf.getFileDuration() / 3600  # in hours
                    channels = edf.getSignalLabels()
                    if Nodes.BTEleft in channels and Nodes.CROSStop in channels:
                        table_subject['SD Configuration'] = 'Left'
                    elif Nodes.BTEright in channels and Nodes.CROSStop in channels:
                        table_subject['SD Configuration'] = 'Right'
                    elif Nodes.BTEleft in channels and Nodes.BTEright in channels:
                        table_subject['SD Configuration'] = 'Generalized'
                    else:
                        print(f"!!! Warning: Unable to determine SD Configuration for subject {subject} in location {location} !!!")


            tsv_files = [f for f in os.listdir(subject_path) if f.endswith('.tsv')]
            for tsv_file in tsv_files:
                # print(f"        | Processing TSV file: {tsv_file}")
                tsv_path = os.path.join(subject_path, tsv_file)
                df = pd.read_csv(tsv_path, delimiter="\t", skiprows=4)
                for i_row, row in df.iterrows():
                    if row['class'] == 'seizure':
                        seizure_duration = row['stop'] - row['start']
                        if 'pediatric' in location.lower():
                            seizure_durations_pediatric.append(seizure_duration)
                        else:
                            seizure_durations_adult.append(seizure_duration)

                        table_subject['Affected Lobe'] = row['localization'].title()
                        if row['type'].lower() == 'fa':
                            table_subject['FA'] += 1
                        elif row['type'].lower() == 'fia':
                            table_subject['FIA'] += 1
                        elif row['type'].lower() == 'fbtc':
                            table_subject['FBTC'] += 1
                        elif row['type'].lower() == 'subclinical':
                            table_subject['Subclinical'] += 1
                        elif row['type'].lower() == 'focal':
                            table_subject['Focal'] += 1
                        elif row['type'].lower() == 'unknown':
                            table_subject['Unknown'] += 1
                        else:
                            print(f"!!! Warning: Unknown seizure type {row['type']} for subject {subject} in location {location} !!!")

                        lateralizations.append(row['lateralization'])

            table.append(table_subject)
    table = pd.DataFrame(table, columns=columns)
    table.to_csv(os.path.join(base_dir, "results", "dataset_content.csv"), index=False)
    print("Dataset content saved to dataset_content.csv")

    print(f"Seizure durations (adult): {np.mean(seizure_durations_adult)} ± {np.std(seizure_durations_adult)} seconds")
    print(f"Seizure durations (pediatric): {np.mean(seizure_durations_pediatric)} ± {np.std(seizure_durations_pediatric)} seconds")
    print(f"Seizure durations (all): {np.mean(seizure_durations_adult + seizure_durations_pediatric)} ± {np.std(seizure_durations_adult + seizure_durations_pediatric)} seconds")
    print(f"Seizure range (all): {min(seizure_durations_adult + seizure_durations_pediatric)} to {max(seizure_durations_adult + seizure_durations_pediatric)} seconds")
    print(f"Lateralizations: {lateralizations.count('left')} left and {lateralizations.count('right')} right. Total: {len(lateralizations)}")
    print(f"Average recording duration: {table['Duration Recordings (hours)'].mean()} ± {table['Duration Recordings (hours)'].std()} hours")
    print(f"Focal aware seizures: {table['FA'].sum()}")
    print(f"Focal impaired awareness seizures: {table['FIA'].sum()}")
    print(f"Affected lobes: {table['Affected Lobe'].value_counts().to_dict()}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["channel_names", "subjects_with_seizures",
                                                   "channel_differences", "rank_locations", "average_memory_size",
                                                   "seizure_segments", "hours_of_data", "compute_dataset_stats",
                                                   "subjects_wo_channels", "find_ref_channel", "dataset_content"],)
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
    elif args.task == "subjects_wo_channels":
        # find_subjects_without_channels(data_path_, channels=Nodes.a_nodes, locations=locations_)
        find_subjects_without_channels(data_path_,
                                       channels=Nodes.baseline_eeg_nodes + Nodes.optional_F_nodes + Nodes.optional_P_nodes + ['T9', 'T10'],
                                       locations=locations_)
    elif args.task == "find_ref_channel":
        find_ref_channel(data_path_, locations=locations_)
    elif args.task == 'dataset_content':
        dataset_content(data_path_, locations=locations_)
    else:
        raise ValueError(f"Unknown task: {args.task}. Use 'channel_names' or 'subjects_with_seizures'.")
