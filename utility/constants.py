import os

from TSelect.tselect.tselect.utils.metrics import auroc_score
from net.DL_config import Config
from net.utils import get_sens_FA_score

SEED = 0

class Nodes:

    basic_eeg_nodes = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    optional_eeg_nodes = ['AF11', 'AF12', 'AF1', 'AF2', 'AF7', 'AF8', 'B2', 'F7', 'F8', 'P7', 'P8', 'F9', 'F10', 'T9',
                          'T10',
                          'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'PO1', 'PO2', 'Iz', 'T7',
                          'T8',
                          'A1', 'A2', 'FT9', 'FT10', 'FT7', 'FT8', 'TP7', 'TP8']

    wearable_nodes = ['BTEleft SD', 'BTEright SD', 'CROSStop SD']
    BTEleft = 'BTEleft SD'
    BTEright = 'BTEright SD'
    CROSStop = 'CROSStop SD'

    eeg_acc = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
    eeg_gyr = ['EEG SD GYR A', 'EEG SD GYR B', 'EEG SD GYR C']

    ecg_emg_nodes = ['ECG+', 'sEMG+', 'ECG SD', 'EMG SD']
    ecg_emg_acc = ['ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z']
    ecg_emg_gyr = ['ECGEMG SD GYR A', 'ECGEMG SD GYR B', 'ECGEMG SD GYR C']

    other_nodes = ['Ment+', 'A1*', 'A2*', 'MKR+', 'B1', 'B2']

    switchable_nodes = {
        BTEright: [BTEleft, CROSStop],
        BTEleft: [BTEright, CROSStop],
        CROSStop: [BTEright, BTEleft]
    }


class Locations:
    coimbra = "Coimbra_University_Hospital"
    freiburg = "Freiburg_University_Medical_Center"
    karolinska = "Karolinska_Institute"
    leuven_adult = "University_Hospital_Leuven_Adult"
    leuven_pediatric = "University_Hospital_Leuven_Pediatric"
    aachen = "University_of_Aachen"

class Keys:
    minirocketLR = "MiniRocketLR"

class Paths:
    remote_data_path = "/cw/dtaidata/ml/2025-Epilepsy"
    local_data_path = "/media/loren/Seagate Basic/Epilepsy use case"  # path to dataset
    remote_save_dir = "/cw/dtailocal/loren/2025-Epilepsy/net/save_dir"
    local_save_dir = "net/save_dir"  # save directory of intermediate and output files


def get_base_config(base_dir, model="ChronoNet", included_channels=None, pretty_name=None, suffix=""):
    """
    Function to get the base configuration for the model. The function sets the parameters for the model, including
    the data path, save directory, sampling frequency, number of channels, batch size, window size, stride, balancing
    factor, validation percentage, and network hyperparameters. The function also sets the model architecture,
    dataset, and sampling method. The function returns a Config object with the specified parameters.
    Args:
        base_dir (str): path to the base directory where the data is stored.
        model (str): model architecture (Options: Chrononet, EEGnet, DeepConvNet, MiniRocketLR)
        included_channels (list): list of channels to include in the model. If None, all channels are included.
        pretty_name (str): pretty name for the experiment.
        suffix (str): suffix to add to the end of the experiment's config name.

    Returns:
        config (Config): Config object with the specified parameters.
    """
    if included_channels is None:
        included_channels = "all"

    if included_channels == "all":
        included_channels = Nodes.basic_eeg_nodes + Nodes.wearable_nodes
    elif included_channels == "wearables":
        included_channels = Nodes.wearable_nodes
        suffix = "wearables" + ("__" if len(suffix) != 0 else "") + suffix
    else:
        raise ValueError(f"Invalid argument for included_channels: {included_channels}. Options are None, 'all' or 'wearables'.")

    config = Config()
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
    config.included_channels = included_channels
    config.CH = len(config.included_channels)  # Nr of EEG channels
    config.cross_validation = 'leave_one_person_out'  # validation type
    config.batch_size = 128  # batch size
    config.frame = 2  # window size of input segments in seconds
    config.stride = 1  # stride between segments (of background EEG) in seconds
    config.stride_s = 0.5  # stride between segments (of seizure EEG) in seconds
    config.boundary = 0.5  # proportion of seizure data in a window to consider the segment in the positive class
    config.factor = 5  # balancing factor between nr of segments in each class
    # proportion of the training set to use for validation
    if model.lower() == Keys.minirocketLR.lower():
        config.validation_percentage = None
    else:
        config.validation_percentage = 0.2
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
    config.locations = [Locations.coimbra]  # locations to use

    ###########################################
    ###########################################
    config.add_to_name = f'{"_" + suffix if suffix != "" else ""}'  # str to add to the end of the experiment's config name

    return config

def get_channel_selection_config(base_dir, model="ChronoNet", included_channels=None, evaluation_metric=auroc_score, auc_percentage=0.6,
                                 corr_threshold=0.5, pretty_name=None, suffix=""):
    config = get_base_config(base_dir, model=model, included_channels=included_channels, pretty_name=pretty_name,
                             suffix=suffix)
    config.channel_selection = True
    config.selected_channels = None
    config.channel_selection_evaluation_metric = evaluation_metric
    config.auc_percentage = auc_percentage
    config.corr_threshold = corr_threshold
    config.add_to_name = (f'{"_channel_selection" if config.channel_selection else ""}'
                          f'{f"_evaluation_metric_{evaluation_metric.__name__}" if evaluation_metric != auroc_score else ""}'
                          f'{f"_auc_percentage_{auc_percentage}" if auc_percentage != 0.6 else ""}'
                            f'{f"_corr_threshold_{corr_threshold}" if corr_threshold != 0.5 else ""}'
                      f'{config.add_to_name}')  # str to add to the end of the experiment's config name

    return config


evaluation_metrics = {"roc_auc": auroc_score,
                      "score": get_sens_FA_score}
