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
parser.add_argument("--nodes", type=str, nargs="?", default="all")
parser.add_argument("--auc", type=float, nargs="?", default=0.6)
parser.add_argument("--corr", type=float, nargs="?", default=0.5)
parser.add_argument("--suffix", type=str, nargs="?", default="")

args = parser.parse_args()

evaluation_metrics = {"roc_auc": auroc_score,
                      "score": get_sens_FA_score}
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
