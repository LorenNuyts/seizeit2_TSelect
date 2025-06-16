import argparse
import os
import random
import shutil

from net.DL_config import get_base_config, get_channel_selection_config
from utility.constants import *
from utility.constants import parse_location
from utility.paths import get_path_results, get_path_config
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

base_ = os.path.dirname(os.path.realpath(__file__))

###########################################
### Parse parameters from command line ####
###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--channel_selection', action='store_true')
parser.add_argument("--model", type=str, nargs="?", default="ChronoNet")
parser.add_argument("--evaluation_metric", type=str, nargs="?", default="roc_auc")
parser.add_argument(
    '--locations',
    nargs='+',  # accept multiple inputs
    type=parse_location,
    default=[Locations.leuven_adult],
    help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
         f"Defaults to [{Locations.leuven_adult}]."
)
parser.add_argument("--nodes", type=str, nargs="?", default="all")
parser.add_argument("--auc", type=float, nargs="?", default=0.6)
parser.add_argument("--corr", type=float, nargs="?", default=0.5)
parser.add_argument("--batch_size", type=int, nargs="?", default=7)
parser.add_argument("--suffix", type=str, nargs="?", default="")
parser.add_argument('--reset', action='store_true')

args = parser.parse_args()

suffix_ = args.suffix
unique_locations = list(dict.fromkeys(args.locations))

############################################
##### Print the settings for this run ######
############################################

print("Locations:", unique_locations)
print("Channel selection:", args.channel_selection)
print("Evaluation metric:", args.evaluation_metric)
print("Model:", args.model)

###########################################
## Initialize standard config parameters ##
###########################################

if args.channel_selection:
    config = get_channel_selection_config(base_, unique_locations, model=args.model,
                                          evaluation_metric=evaluation_metrics[args.evaluation_metric],
                                          auc_percentage=args.auc, corr_threshold=args.corr,
                                          suffix=suffix_, included_channels=args.nodes, batch_size=args.batch_size)
else:
    config = get_base_config(base_, unique_locations, model=args.model, suffix=suffix_, included_channels=args.nodes,
                             batch_size=args.batch_size)

###########################################
###########################################
config_path = get_path_config(config, config.get_name())
results_path = get_path_results(config, config.get_name())

if args.reset:
    print('Deleting results files...')
    if os.path.exists(results_path):
        os.remove(results_path)
    second_result = results_path.replace('__all_results.pkl', '.h5')
    if os.path.exists(second_result):
        os.remove(second_result)
    print('Deleting models folder...')
    model_path = os.path.join(config.save_dir, 'models', config.get_name())
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    print("Deleting predictions folder...")
    predictions_path = os.path.join(config.save_dir, 'predictions', config.get_name())
    if os.path.exists(predictions_path):
        shutil.rmtree(predictions_path)

if os.path.exists(config_path):
    config.load_config(config_path, config.get_name())
    # Loading the results from barabas on my personal computer
    if 'dtai' in config.save_dir and 'dtai' not in base_:
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)


##### RESULTS:
results = Results(config)
results_path = get_path_results(config, config.get_name())
if os.path.exists(results_path):
  results.load_results(results_path)
  # Loading the results from barabas on my personal computer
  if 'dtai' in results.config.save_dir and 'dtai' not in base_:
    results.config.save_dir = results.config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
    results.config.data_path = results.config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
  config = results.config # If the config file is not there (because I didn't download it), load the config from the results file

###########################################
###########################################
load_generators = False                                          # Boolean to load generators from file
save_generators = False                                         # Boolean to save the training and validation generator objects. The training generator is saved with the dataset, frame and sample type properties in the name of the file. The validation generator is always using the sequential windowed method.


main_func.train(config, results, load_generators, save_generators)

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config, results)
