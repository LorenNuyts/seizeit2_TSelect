import os
import pickle
from typing import Dict, List

import numpy as np

from net.DL_config import Config


class Results:
    def __init__(self, config: Config):
        self.config: Config = config
        self.selection_time: Dict[int, float] = {}
        self.train_time: Dict[int, float] = {}
        self.f1_ovlp: List[list] = []
        self.fah_ovlp: List[list] = []
        self.prec_ovlp: List[list] = []
        self.sens_ovlp: List[list] = []
        self.score: List[list] = []
        self.thresholds: list = []

    @property
    def total_time(self) -> Dict[int, float]:
        total_time = {}
        for fold in self.train_time.keys():
            selection_time = self.train_time[fold] if fold in self.selection_time.keys() else 0
            total_time[fold] = selection_time + self.train_time[fold]
        return total_time

    @property
    def average_selection_time(self) -> float:
        return sum(self.selection_time.values()) / len(self.selection_time)

    @property
    def average_train_time(self) -> float:
        return sum(self.train_time.values()) / len(self.train_time)

    @property
    def average_total_time(self) -> float:
        return sum(self.total_time.values()) / len(self.total_time)

    @property
    def average_f1_ovlp_all_thresholds(self) -> List[float]:
        return [sum(f1) / len(f1) for f1 in zip(*self.f1_ovlp)]

    @property
    def average_fah_ovlp_all_thresholds(self) -> List[float]:
        return [sum(fah) / len(fah) for fah in zip(*self.fah_ovlp)]

    @property
    def average_prec_ovlp_all_thresholds(self) -> List[float]:
        return [sum(prec) / len(prec) for prec in zip(*self.prec_ovlp)]

    @property
    def average_sens_ovlp_all_thresholds(self) -> List[float]:
        return [sum(sens) / len(sens) for sens in zip(*self.sens_ovlp)]

    @property
    def average_score_all_thresholds(self) -> List[float]:
        return [sum(score) / len(score) for score in zip(*self.score)]

    @property
    def best_average_f1_ovlp(self) -> (float, float):
        best_value = max(self.average_f1_ovlp_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_f1_ovlp_all_thresholds.index(best_value)]

    @property
    def best_average_fah_ovlp(self) -> (float, float):
        best_value = max(self.average_fah_ovlp_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_fah_ovlp_all_thresholds.index(best_value)]

    @property
    def best_average_prec_ovlp(self) -> (float, float):
        best_value = max(self.average_prec_ovlp_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_prec_ovlp_all_thresholds.index(best_value)]

    @property
    def best_average_sens_ovlp(self) -> (float, float):
        best_value = max(self.average_sens_ovlp_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_sens_ovlp_all_thresholds.index(best_value)]

    @property
    def best_average_score(self) -> (float, float):
        best_value = max(self.average_score_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_score_all_thresholds.index(best_value)]

    @property
    def average_f1_ovlp(self) -> np.float32:
        return np.mean(self.average_f1_ovlp_all_thresholds)

    @property
    def average_fah_ovlp(self) -> np.float32:
        return np.mean(self.average_fah_ovlp_all_thresholds)

    @property
    def average_prec_ovlp(self) -> np.float32:
        return np.mean(self.average_prec_ovlp_all_thresholds)

    @property
    def average_sens_ovlp(self) -> np.float32:
        return np.mean(self.average_sens_ovlp_all_thresholds)

    @property
    def average_score(self) -> np.float32:
        return np.mean(self.average_score_all_thresholds)

    @property
    def name(self) -> str:
        return self.config.get_name()

    def save_results(self, save_path):
        with open(save_path, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)


    def load_results(self, results_path):
        if not os.path.exists(results_path):
            raise ValueError('Directory is empty or does not exist')
        with open(results_path, 'rb') as input:
            config = pickle.load(input)

        self.__dict__.update(config)