from collections import defaultdict
from typing import List

import tensorflow as tf
import pickle
import os

from TSelect.tselect.tselect.utils.metrics import auroc_score
from utility.constants import Locations, Nodes, Paths, Keys, evaluation_metrics

CURRENT_VERSION = 2

class Config():
    """ Class to create and store an experiment configuration object with the architecture hyper-parameters, input and sampling types.
    
    Args:
        data_path (str): path to data
        model (str): model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
        dataset (str): patients split  (check 'datasets' folder)

        fs (int): desired sampling frequency of the input data.
        CH (int): number of channels of the input data.
        frame (int): window size of input segments in seconds.
        stride (float): stride between segments (of background EEG) in seconds
        stride_s (float): stride between segments (of seizure EEG) in seconds
        boundary (float): proportion of seizure data in a window to consider the segment in the positive class
        batch_size (int): batch size for training model
        sample_type (str): sampling method (default is subsample, removes background EEG segments to match the number of seizure segments times the balancing factor)
        factor(int): balancing factor between number of segments in each class. The number of background segments is the number of seizure segments times the balancing factor.
        l2 (float): L2 regularization penalty
        lr (float): learning rate
        dropoutRate (float): layer's dropout rate
        nb_epochs (int): number of epochs to train model
        class_weights (dict): weight of each class for computing the loss function
        cross_validation (str): validation type (default is 'fixed' set of patients for training and validation)
        save_dir (str): save directory for intermediate and output files

    """

    def __init__(self, data_path=None, model='ChronoNet', dataset='SZ2', fs=None, CH=None, frame=2, stride=1,
                 stride_s=0.5, boundary=0.5, batch_size=64, sample_type='subsample', factor=5, l2=0, lr=0.01,
                 dropoutRate=0, nb_epochs=50, class_weights = {0:1, 1:1}, cross_validation=Keys.stratified, save_dir='savedir',
                 held_out_fold = False, Fz_reference=False, version_experiments=CURRENT_VERSION):

        self.random_seed = 0
        self.data_path = data_path
        self.model = model
        self.dataset = dataset
        self.save_dir = save_dir
        self.fs = fs
        self.CH = CH
        self.frame = frame
        self.stride = stride
        self.stride_s = stride_s
        self.boundary = boundary
        self.batch_size = batch_size
        self.sample_type = sample_type
        self.factor = factor
        self.cross_validation = cross_validation
        self.save_dir = save_dir
        self.locations: List[str] = []
        self.included_channels = None
        self.channel_selection = False
        self.selected_channels = None
        self.channel_selection_settings = None
        self.channel_selector = defaultdict()  # dictionary to store the channel selector for each fold
        self.pretty_name = None
        self.folds = {}  # dictionary to store the folds
        self.n_folds = 10  # number of folds for cross-validation
        self.held_out_fold = held_out_fold
        self.held_out_subjects = None
        self.version_experiments = version_experiments
        self.Fz_reference = Fz_reference

        # models parameters
        self.data_format = tf.keras.backend.image_data_format
        self.l2 = l2
        self.lr = lr
        self.dropoutRate = dropoutRate
        self.nb_epochs = nb_epochs
        self.class_weights = class_weights

    def reload_CH(self, fold=None):
        if self.channel_selection and fold is not None:
            self.CH = len(self.selected_channels[fold])
        elif self.channel_selection:
            self.CH = len(self.included_channels)

    def save_config(self, save_path):
        name = self.get_name()
        with open(os.path.join(save_path, name + '.cfg'), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
            print("Config saved:", os.path.join(save_path, name + '.cfg'))


    def load_config(self, config_path, config_name):
        if not os.path.exists(config_path):
            raise ValueError('Directory is empty or does not exist')

        with open(os.path.join(config_path, config_name + '.cfg'), 'rb') as input:
            config = pickle.load(input)

        self.__dict__.update(config)

        
    def get_name(self):
        locations_str = "-".join([Locations.to_acronym(loc) for loc in self.locations])
        base_name = '_'.join([self.model, self.sample_type, 'factor' + str(self.factor), self.cross_validation + "CV",])
        if self.held_out_fold:
            base_name += '_held_out_fold'
        if locations_str != "":
            base_name += '_' + locations_str
        # if set(self.included_channels).intersection(set(Nodes.wearable_nodes)) == 0:
        #     base_name += '_no_wearables'
        if self.Fz_reference:
            base_name += '_Fz_ref'
        if hasattr(self, 'add_to_name') and self.add_to_name != "":
            base_name = base_name + '_' + self.add_to_name
        if self.version_experiments is not None:
            base_name += f'_v{self.version_experiments}'
        return base_name


def get_base_config(base_dir, locations, model="ChronoNet", batch_size=128,
                    included_channels=None, CV=Keys.stratified, held_out_fold=False, pretty_name=None,
                    Fz_reference=False,
                    version_experiments=CURRENT_VERSION, suffix=""):
    """
    Function to get the base configuration for the model. The function sets the parameters for the model, including
    the data path, save directory, sampling frequency, number of channels, batch size, window size, stride, balancing
    factor, validation percentage, and network hyperparameters. The function also sets the model architecture,
    dataset, and sampling method. The function returns a Config object with the specified parameters.
    Args:
        base_dir (str): path to the base directory where the data is stored.
        locations (list): list of locations to use for the experiment.
        model (str): model architecture (Options: Chrononet, EEGnet, DeepConvNet, MiniRocketLR)
        batch_size (int): batch size for training the model.
        included_channels (str): list of channels to include in the model. If None, all channels are included.
        CV (str): cross-validation method to use. Options are 'leave_one_person_out' or 'stratified'.
        held_out_fold (bool): whether to use a held-out fold that is not used for training, validation, or testing.
        pretty_name (str): pretty name for the experiment.
        suffix (str): suffix to add to the end of the experiment's config name.

    Returns:
        config (Config): Config object with the specified parameters.
    """
    if included_channels is None:
        included_channels = "all"

    if included_channels == "all":
        included_channels = Nodes.basic_eeg_nodes + Nodes.included_wearables
    elif included_channels == "wearables":
        included_channels = Nodes.included_wearables
        suffix = "wearables" + ("__" if len(suffix) != 0 else "") + suffix
    elif included_channels == "no_wearables":
        included_channels = Nodes.basic_eeg_nodes
        suffix = "no_wearables" + ("__" if len(suffix) != 0 else "") + suffix
    elif included_channels == "Cross_T7":
        included_channels = [Nodes.CROSStop, "T7"]
        suffix = "Cross_T7" + ("__" if len(suffix) != 0 else "") + suffix
    elif included_channels == "T7":
        included_channels = ["T7"]
        suffix = "T7" + ("__" if len(suffix) != 0 else "") + suffix
    elif included_channels == "CROSStop":
        included_channels = [Nodes.CROSStop]
        suffix = "CROSStop" + ("__" if len(suffix) != 0 else "") + suffix
    elif included_channels.startswith("[") and included_channels.endswith("]"):
        # Parse the string representation of the list
        included_channels_str = included_channels[1:-1].replace(",", "_").strip()
        included_channels = included_channels.strip("[]").split(",")
        included_channels = [ch.strip().strip("'").strip('"') for ch in included_channels]

        # Replace CROSStop, BTEleft, BTEright with their respective names
        included_channels = [Nodes.CROSStop if ch == "CROSStop" else ch for ch in included_channels]
        included_channels = [Nodes.BTEleft if ch == "BTEleft" else ch for ch in included_channels]
        included_channels = [Nodes.BTEright if ch == "BTEright" else ch for ch in included_channels]
        suffix = included_channels_str + ("__" if len(suffix) != 0 else "") + suffix
    else:
        raise ValueError(f"Invalid argument for included_channels: {included_channels}. Options are None, 'all', 'wearables',"
                         f"'Cross_T7' or 'T7'.")

    config = Config(model=model, batch_size=batch_size, cross_validation=CV, held_out_fold=held_out_fold,
                    version_experiments=version_experiments, Fz_reference=Fz_reference)
    if pretty_name:
        config.pretty_name = pretty_name
    if 'dtai' in base_dir:
        config.data_path = Paths.remote_data_path
        config.save_dir = Paths.remote_save_dir
    else:
        config.data_path = Paths.local_data_path  # path to dataset
        config.save_dir = Paths.local_save_dir
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    config.fs = 250  # Sampling frequency of the data after post-processing
    config.included_channels = sorted(included_channels)
    config.CH = len(config.included_channels)  # Nr of EEG channels
    if Nodes.BTEleft in config.included_channels and Nodes.BTEright in config.included_channels and \
        Nodes.CROSStop in config.included_channels:
        config.CH -= 1  # Only two of the three wearable channels can be used at the same time,
                        # so we reduce the number of channels by 1

    config.cross_validation = CV  # validation type
    config.batch_size = batch_size  # batch size for the training set
    config.val_batch_size = 6 * 60  # batch size for the validation set
    config.test_batch_size = 100 * 6 * 60 # batch size for the test set
    config.frame = 2  # window size of input segments in seconds
    config.stride = 1  # stride between segments (of background EEG) in seconds
    config.stride_s = 0.5  # stride between segments (of seizure EEG) in seconds
    config.boundary = 0.5  # proportion of seizure data in a window to consider the segment in the positive class
    config.factor = 5  # balancing factor between nr of segments in each class
    config.train_percentage = 0.8  # proportion of the dataset to use for training
    config.validation_percentage = 0.1  # proportion of the dataset to use for the validation set
    config.n_folds = 10  # number of folds for cross-validation
    config.folds = {}  # dictionary to store the folds

    ## Network hyper-parameters
    config.dropoutRate = 0.5
    config.nb_epochs = 300
    config.l2 = 0.01
    config.lr = 0.01

    ###########################################
    ###########################################

    ##### INPUT CONFIGS:
    config.model = model  # model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
    config.dataset = 'SZ2'  # patients to use (check 'datasets' folder)
    config.sample_type = 'subsample'  # sampling method (subsample = remove background EEG segments)

    ###########################################
    ###########################################

    ##### DATA CONFIGS:
    config.locations = locations  # locations to use

    ###########################################
    ###########################################
    config.add_to_name = f'{"_" + suffix if suffix != "" else ""}'  # str to add to the end of the experiment's config name

    return config


def get_channel_selection_config(base_dir, locations, model="ChronoNet", batch_size=128,
                                 included_channels=None, evaluation_metric=evaluation_metrics['score'],
                                 irrelevant_selector_percentage=0.6,
                                 corr_threshold=0.5, irrelevant_selector_threshold=-100, CV=Keys.stratified,
                                 held_out_fold=False,
                                 pretty_name=None, Fz_reference=False,
                                version_experiments=CURRENT_VERSION, suffix="") -> Config:
    config = get_base_config(base_dir, locations, model=model, included_channels=included_channels,
                             pretty_name=pretty_name, batch_size=batch_size, CV=CV, held_out_fold=held_out_fold,
                             version_experiments=version_experiments, Fz_reference=Fz_reference,
                             suffix=suffix)
    config.channel_selection = True
    config.selected_channels = None
    config.channel_selection_settings = {
        'evaluation_metric': evaluation_metric,
        'irrelevant_selector_percentage': irrelevant_selector_percentage,
        'corr_threshold': corr_threshold,
        'irrelevant_selector_threshold': irrelevant_selector_threshold,
    }
    config.add_to_name = (f'{"_channel_selection" if config.channel_selection else ""}'
                          f'{f"_evaluation_metric_{evaluation_metric.__name__}" if evaluation_metric != auroc_score else ""}'
                          f'{f"_irr_th_{int(irrelevant_selector_threshold*100)}" if irrelevant_selector_threshold != 0.5 else ""}'
                          f'{f"_auc_percentage_{int(irrelevant_selector_percentage * 100)}" if irrelevant_selector_percentage != 0.6 else ""}'
                            f'{f"_corr_threshold_{int(corr_threshold*100)}" if corr_threshold != 0.5 else ""}'
                      f'{config.add_to_name}')  # str to add to the end of the experiment's config name

    return config
