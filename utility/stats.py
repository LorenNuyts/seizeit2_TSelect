import os
import pickle
from typing import Dict, List, Any

import numpy as np
from numpy import floating

from net.DL_config import Config


class Results:
    def __init__(self, config: Config):
        self.config: Config = config
        self.selection_time: Dict[int, float] = {}
        self.train_time: Dict[int, float] = {}
        self.f1_ovlp: List[list] = []
        self.fah_ovlp: List[list] = []
        self.fah_epoch: List[list] = []
        self.prec_ovlp: List[list] = []
        self.sens_ovlp: List[list] = []
        self.rocauc: List[list] = []
        self.score: List[list] = []
        self.thresholds: list = []

    @property
    def total_time(self) -> Dict[int, float]:
        total_time = {}
        for fold in self.train_time.keys():
            selection_time = self.selection_time[fold] if fold in self.selection_time.keys() else 0
            total_time[fold] = selection_time + self.train_time[fold]
        return total_time

    @property
    def average_selection_time(self) -> float:
        if len(self.selection_time) == 0:
            return 0
        return sum(self.selection_time.values()) / len(self.selection_time)

    @property
    def std_selection_time(self) -> float | floating[Any]:
        if len(self.selection_time) == 0:
            return 0
        return np.nanstd(list(self.selection_time.values()))

    @property
    def average_train_time(self) -> float:
        if len(self.train_time) == 0:
            return 0
        return sum(self.train_time.values()) / len(self.train_time)

    @property
    def std_train_time(self) -> float | floating[Any]:
        if len(self.train_time) == 0:
            return 0
        return np.nanstd(list(self.train_time.values()))

    @property
    def average_total_time(self) -> float:
        if len(self.total_time) == 0:
            return 0
        return sum(self.total_time.values()) / len(self.total_time)

    @property
    def std_total_time(self) -> float | floating[Any]:
        if len(self.total_time) == 0:
            return 0
        return np.nanstd(list(self.total_time.values()))

    @property
    def nb_channels(self) -> List[int]:
        return [len(chs) for chs in
                           self.config.selected_channels.values()] if self.config.channel_selection else self.config.CH

    @property
    def average_nb_channels(self) -> float:
        return np.nanmean([len(chs) for chs in self.config.selected_channels.values()]) if self.config.channel_selection else self.config.CH

    @property
    def std_nb_channels(self) -> float:
        return np.nanstd([len(chs) for chs in self.config.selected_channels.values()]) if self.config.channel_selection else 0

    @property
    def average_f1_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(f1) for f1 in zip(*self.f1_ovlp)]

    @property
    def std_f1_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(f1) for f1 in zip(*self.f1_ovlp)]

    @property
    def average_fah_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(fah) for fah in zip(*self.fah_ovlp)]

    @property
    def std_fah_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(fah) for fah in zip(*self.fah_ovlp)]

    @property
    def average_fah_epoch_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(fah) for fah in zip(*self.fah_epoch)]

    @property
    def std_fah_epoch_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(fah) for fah in zip(*self.fah_epoch)]

    @property
    def average_prec_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(prec) for prec in zip(*self.prec_ovlp)]

    @property
    def std_prec_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(prec) for prec in zip(*self.prec_ovlp)]

    @property
    def average_sens_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(sens) for sens in zip(*self.sens_ovlp)]

    @property
    def std_sens_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(sens) for sens in zip(*self.sens_ovlp)]

    @property
    def average_rocauc_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(score) for score in zip(*self.rocauc)]

    @property
    def std_rocauc_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(score) for score in zip(*self.rocauc)]

    @property
    def average_score_all_thresholds(self) -> List[np.float32]:
        return [np.nanmean(score) for score in zip(*self.score)]

    @property
    def std_score_all_thresholds(self) -> List[np.float32]:
        return [np.nanstd(score) for score in zip(*self.score)]

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
    def best_average_fah_epoch(self) -> (float, float):
        best_value = max(self.average_fah_epoch_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_fah_epoch_all_thresholds.index(best_value)]

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
    def best_average_rocauc(self) -> (float, float):
        best_value = max(self.average_rocauc_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_rocauc_all_thresholds.index(best_value)]

    @property
    def best_average_score(self) -> (float, float):
        if len(self.average_score_all_thresholds) == 0:
            return np.nan, np.nan
        best_value = max(self.average_score_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.average_score_all_thresholds.index(best_value)]

    @property
    def average_f1_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_f1_ovlp_all_thresholds[index_best_threshold]

    @property
    def average_fah_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_fah_ovlp_all_thresholds[index_best_threshold]

    @property
    def average_fah_epoch_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_fah_epoch_all_thresholds[index_best_threshold]

    @property
    def average_prec_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_prec_ovlp_all_thresholds[index_best_threshold]

    @property
    def average_sens_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_sens_ovlp_all_thresholds[index_best_threshold]

    @property
    def average_rocauc_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_rocauc_all_thresholds[index_best_threshold]

    @property
    def average_score_best_threshold(self) -> np.float32:
        best_threshold = self.best_average_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.average_score_all_thresholds[index_best_threshold]

    @property
    def average_f1_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_f1_ovlp_all_thresholds[index_th05]

    @property
    def average_fah_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_fah_ovlp_all_thresholds[index_th05]

    @property
    def average_fah_epoch_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_fah_epoch_all_thresholds[index_th05]

    @property
    def average_prec_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_prec_ovlp_all_thresholds[index_th05]

    @property
    def average_sens_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_sens_ovlp_all_thresholds[index_th05]

    @property
    def average_rocauc_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_rocauc_all_thresholds[index_th05]

    @property
    def average_score_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.average_score_all_thresholds[index_th05]

    @property
    def std_f1_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_f1_ovlp_all_thresholds[index_best_threshold]

    @property
    def std_fah_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_fah_ovlp_all_thresholds[index_best_threshold]

    @property
    def std_fah_epoch_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_fah_epoch_all_thresholds[index_best_threshold]

    @property
    def std_prec_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_prec_ovlp_all_thresholds[index_best_threshold]

    @property
    def std_sens_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_sens_ovlp_all_thresholds[index_best_threshold]

    @property
    def std_rocauc_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_rocauc_all_thresholds[index_best_threshold]

    @property
    def std_score_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.std_score_all_thresholds[index_best_threshold]

    @property
    def std_f1_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_f1_ovlp_all_thresholds[index_th05]

    @property
    def std_fah_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_fah_ovlp_all_thresholds[index_th05]

    @property
    def std_fah_epoch_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_fah_epoch_all_thresholds[index_th05]

    @property
    def std_prec_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_prec_ovlp_all_thresholds[index_th05]

    @property
    def std_sens_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_sens_ovlp_all_thresholds[index_th05]

    @property
    def std_rocauc_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_rocauc_all_thresholds[index_th05]

    @property
    def std_score_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.std_score_all_thresholds[index_th05]

    @property
    def median_f1_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(f1) for f1 in zip(*self.f1_ovlp)]

    @property
    def median_fah_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(fah) for fah in zip(*self.fah_ovlp)]

    @property
    def median_fah_epoch_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(fah) for fah in zip(*self.fah_epoch)]

    @property
    def median_prec_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(prec) for prec in zip(*self.prec_ovlp)]

    @property
    def median_sens_ovlp_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(sens) for sens in zip(*self.sens_ovlp)]

    @property
    def median_rocauc_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(score) for score in zip(*self.rocauc)]

    @property
    def median_score_all_thresholds(self) -> List[np.float32]:
        return [np.nanmedian(score) for score in zip(*self.score)]

    @property
    def best_median_score(self) -> (float, float):
        if len(self.median_score_all_thresholds) == 0:
            return np.nan, np.nan
        best_value = max(self.median_score_all_thresholds)
        if np.isnan(best_value):
            return np.nan, np.nan
        return best_value, self.thresholds[self.median_score_all_thresholds.index(best_value)]

    @property
    def median_f1_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_f1_ovlp_all_thresholds[index_best_threshold]

    @property
    def median_fah_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_fah_ovlp_all_thresholds[index_best_threshold]

    @property
    def median_fah_epoch_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_fah_epoch_all_thresholds[index_best_threshold]

    @property
    def median_prec_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_prec_ovlp_all_thresholds[index_best_threshold]

    @property
    def median_sens_ovlp_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_sens_ovlp_all_thresholds[index_best_threshold]

    @property
    def median_rocauc_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_rocauc_all_thresholds[index_best_threshold]

    @property
    def median_score_best_threshold(self) -> np.float32:
        best_threshold = self.best_median_score[1]
        if np.isnan(best_threshold):
            return np.float32(np.nan)
        index_best_threshold = self.thresholds.index(best_threshold)
        return self.median_score_all_thresholds[index_best_threshold]

    @property
    def median_f1_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_f1_ovlp_all_thresholds[index_th05]

    @property
    def median_fah_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_fah_ovlp_all_thresholds[index_th05]

    @property
    def median_fah_epoch_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_fah_epoch_all_thresholds[index_th05]

    @property
    def median_prec_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_prec_ovlp_all_thresholds[index_th05]

    @property
    def median_sens_ovlp_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_sens_ovlp_all_thresholds[index_th05]

    @property
    def median_rocauc_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_rocauc_all_thresholds[index_th05]

    @property
    def median_score_th05(self) -> np.float32:
        if 0.5 not in self.thresholds:
            return np.float32(np.nan)
        index_th05 = self.thresholds.index(0.5)
        return self.median_score_all_thresholds[index_th05]

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
            results = pickle.load(input)

        self.__dict__.update(results)