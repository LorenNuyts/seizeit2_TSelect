import argparse
import random
import shutil

from utility.constants import *
from utility.constants import evaluation_metrics
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
parser.add_argument("--evaluation_metric", type=str, nargs="?", default="roc_auc")
parser.add_argument("--nodes", type=str, nargs="?", default="all")
parser.add_argument("--auc", type=float, nargs="?", default=0.6)
parser.add_argument("--corr", type=float, nargs="?", default=0.5)
parser.add_argument("--suffix", type=str, nargs="?", default="")
parser.add_argument('--reset', action='store_true')

args = parser.parse_args()

suffix_ = args.suffix

###########################################
## Initialize standard config parameters ##
###########################################

if args.channel_selection:
    config = get_channel_selection_config(base_, evaluation_metric=evaluation_metrics[args.evaluation_metric],
                                          auc_percentage=args.auc, corr_threshold=args.corr,
                                          suffix=suffix_, included_channels=args.nodes)
else:
    config = get_base_config(base_, suffix=suffix_, included_channels=args.nodes)

###########################################
###########################################
config_path = get_path_config(config, config.get_name())
if os.path.exists(config_path):
    config.load_config(config_path, config.get_name())

##### RESULTS:
results = Results(config)
results_path = get_path_results(config, config.get_name())
if os.path.exists(results_path):
  results.load_results(results_path)

###########################################
###########################################
load_generators = False                                          # Boolean to load generators from file
save_generators = False                                         # Boolean to save the training and validation generator objects. The training generator is saved with the dataset, frame and sample type properties in the name of the file. The validation generator is always using the sequential windowed method.

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

main_func.train(config, results, load_generators, save_generators)

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config, results)
