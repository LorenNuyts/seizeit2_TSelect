import argparse
import os

from net.DL_config import get_base_config
from net.key_generator import generate_data_keys_sequential_window, generate_data_keys_sequential
from utility.constants import Locations

base_ = os.path.dirname(os.path.realpath(__file__))

def negative_dimensions():
    config = get_base_config(os.path.join(base_, ".."), locations=[Locations.karolinska, Locations.freiburg])
    val_recs_list = [#['Karolinska_Institute', 'SUBJ-6-430', 'r26'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r1'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r2'],
                     ['Freiburg_University_Medical_Center', 'SUBJ-4-230', 'r3'],
                     ]
    val_segments = generate_data_keys_sequential(config, val_recs_list, 6 * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    negative_dimensions()