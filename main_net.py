import argparse
import os
import random
import shutil

from net.utils import get_sens_FA_score
from TSelect.tselect.tselect.utils.metrics import auroc_score
from utility.constants import *
from utility.paths import get_path_results
from utility.stats import Results

random_seed = 1
random.seed(random_seed)

import numpy as np
np.random.seed(random_seed)

import tensorflow as tf
tf.random.set_seed(random_seed)

from net import key_generator
key_generator.random.seed(random_seed)

from net import main_func

from net.DL_config import Config

base_ = os.path.dirname(os.path.realpath(__file__))

###########################################
### Parse parameters from command line ####
###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--channel_selection', action='store_true')
parser.add_argument("--evaluation_metric", type=str, nargs="?", default="roc_auc")
parser.add_argument("--suffix", type=str, nargs="?", default="")

args = parser.parse_args()

evaluation_metrics = {"roc_auc": auroc_score,
                      "score": get_sens_FA_score}

###########################################
## Initialize standard config parameters ##
###########################################

if args.channel_selection:
    config = get_channel_selection_config(base_, evaluation_metric=evaluation_metrics[args.evaluation_metric], suffix=args.suffix)
else:
    config = get_base_config(base_, suffix=args.suffix)

# ## Configuration for the generator and models:
# config = Config()
# # config.data_path = '/esat/biomeddata/SeizeIT2/bids'             # path to data
# if 'dtai' in base_:
#   config.data_path = '/cw/dtaidata/ml/2025-Epilepsy'
#   config.save_dir = '/cw/dtailocal/loren/2025-Epilepsy/net/save_dir'
# else:
#   config.data_path = "/media/loren/Seagate Basic/Epilepsy use case"       # path to dataset
#   config.save_dir = 'net/save_dir'                                # save directory of intermediate and output files
# if not os.path.exists(config.save_dir):
#   os.makedirs(config.save_dir)
#
# config.fs = 250                                                 # Sampling frequency of the data after post-processing
# config.included_channels = Nodes.basic_eeg_nodes + Nodes.wearable_nodes
# config.CH = len(config.included_channels)                       # Nr of EEG channels
# config.cross_validation = 'leave_one_person_out'                # validation type TODO: Implement leave one seizure out
# config.batch_size = 128                                         # batch size
# config.frame = 2                                                # window size of input segments in seconds
# config.stride = 1                                               # stride between segments (of background EEG) in seconds
# config.stride_s = 0.5                                           # stride between segments (of seizure EEG) in seconds
# config.boundary = 0.5                                           # proportion of seizure data in a window to consider the segment in the positive class
# config.factor = 5                                               # balancing factor between nr of segments in each class
# config.validation_percentage = 0.2                             # proportion of the training set to use for validation
# config.folds = {}                                               # dictionary to store the folds
#
# ## Network hyper-parameters
# config.dropoutRate = 0.5
# config.nb_epochs = 300
# config.l2 = 0.01
# config.lr = 0.01
#
# ###########################################
# ###########################################
#
# ##### INPUT CONFIGS:
# config.model = 'ChronoNet'                                      # model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
# config.dataset = 'SZ2'                                          # patients to use (check 'datasets' folder)
# config.sample_type = 'subsample'                                # sampling method (subsample = remove background EEG segments)
#
# ###########################################
# ###########################################
#
# ##### DATA CONFIGS:
# config.locations = [Locations.coimbra]              # locations to use
#
# ###########################################
# ###########################################
#
# ##### CHANNEL SELECTION CONFIGS:
# config.channel_selection = args.channel_selection       # whether to use channel selection
# config.selected_channels = None                         # selected channels (if None, no channel selection has been performed yet)
# config.channel_selection_evaluation_metric = evaluation_metrics[args.evaluation_metric]  # evaluation metric to use for channel selection
#
# ###########################################
# ###########################################
#
# # config.add_to_name = 'test'                                     # str to add to the end of the experiment's config name
# config.add_to_name = (f'{"_channel_selection" if config.channel_selection else ""}'
#                       f'{"_" + args.suffix if args.suffix != "" else ""}')  # str to add to the end of the experiment's config name

###########################################
###########################################

##### RESULTS:
results = Results(config)
results_path = get_path_results(config, config.get_name())
if os.path.exists(results_path):
  results.load_results(results_path)

###########################################
###########################################
load_generators = False                                          # Boolean to load generators from file
save_generators = False                                         # Boolean to save the training and validation generator objects. The training generator is saved with the dataset, frame and sample type properties in the name of the file. The validation generator is always using the sequential windowed method.

# if os.path.exists(config.save_dir):
#   shutil.rmtree(config.save_dir)

main_func.train(config, results, load_generators, save_generators)

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config, results)
