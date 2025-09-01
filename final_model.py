import argparse
import random
import shutil
from collections import defaultdict

from net.DL_config import get_base_config, get_channel_selection_config
from utility.constants import *
from utility.constants import parse_location
from utility.paths import *
from utility.stats import Results

random_seed = 1
random.seed(random_seed)

import numpy as np
np.random.seed(random_seed)

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
parser.add_argument("--evaluation_metric", type=str, nargs="?", default="score",
                    choices=list(evaluation_metrics.keys()))
parser.add_argument(
    '--locations',
    nargs='+',  # accept multiple inputs
    type=parse_location,
    default=[parse_location(l) for l in Locations.all_keys()],
    help=f"List of locations. Choose from: {', '.join(Locations.all_keys())}. "
         f"Defaults to all locations."
)
parser.add_argument("--nodes", type=str, nargs="?", default="all")
parser.add_argument("--irr_th", type=float, nargs="?", default=None,)
parser.add_argument("--auc", type=float, nargs="?", default=0.6)
parser.add_argument("--corr", type=float, nargs="?", default=0.5)
parser.add_argument("--batch_size", type=int, nargs="?", default=128)
parser.add_argument("--suffix", type=str, nargs="?", default="")
parser.add_argument('--reset', action='store_true')
parser.add_argument("--gpu", type=int, nargs="?", default=0)
parser.add_argument("--CV", type=str, nargs="?", default=Keys.stratified,
                    choices=["leave_one_person_out", "stratified", "leave_one_hospital_out"],
                    help="Cross-validation method to use. Defaults to 'leave_one_person_out'.")

args = parser.parse_args()

if args.irr_th is None:
    args.irr_th = -100  if args.evaluation_metric=='score' else 0.5

suffix_ = args.suffix
unique_locations = sorted(list(dict.fromkeys(args.locations)))

############################################
##### Set the GPU and import tensorflow ####
############################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import tensorflow as tf
tf.random.set_seed(random_seed)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

############################################
##### Print the settings for this run ######
############################################

print("Locations:", unique_locations)
print("Model:", args.model)

###########################################
## Initialize standard config parameters ##
###########################################

config = get_base_config(base_, unique_locations, model=args.model, suffix=suffix_ + "_final_model_" + args.nodes,
                         included_channels=args.nodes,
                             batch_size=args.batch_size)
dual_config = get_channel_selection_config(base_, unique_locations, model=args.model,
                                          evaluation_metric=evaluation_metrics[args.evaluation_metric],
                                          irrelevant_selector_threshold=args.irr_th,
                                          irrelevant_selector_percentage=args.auc, corr_threshold=args.corr, CV=args.CV,
                                          suffix=suffix_, included_channels='all', batch_size=args.batch_size,
                                          held_out_fold=True)

###########################################
###########################################
config_path = get_path_config(config, config.get_name())
dual_config_path = get_path_config(dual_config, dual_config.get_name())
print('Config path:', config_path)
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
    # print("Deleting generators folder...")
    # generators_path = os.path.join(config.save_dir, 'generators', config.get_name())
    # if os.path.exists(generators_path):
    #     shutil.rmtree(generators_path)
    print("Deleting precomputed segments folder...")
    segments_path = os.path.join(config.save_dir, 'segments', config.get_name())
    if os.path.exists(segments_path):
        shutil.rmtree(segments_path)

if os.path.exists(config_path):
    config.load_config(config_path, config.get_name())
    # Loading the results from barabas on my personal computer
    if 'dtai' in config.save_dir and 'dtai' not in base_:
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)

dual_config.load_config(dual_config_path, dual_config.get_name())

##### RESULTS:
results = Results(config)
results_path = get_path_results(config, config.get_name())
if os.path.exists(results_path):
    results.load_results(results_path)
    # Loading the results from barabas on my personal computer
    if 'dtai' in results.config.save_dir and 'dtai' not in base_:
        results.config.save_dir = results.config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        results.config.data_path = results.config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
    if not os.path.exists(config_path):
        config = results.config # If the config file is not there (because I didn't download it), load the config from the results file

config.held_out_subjects = dual_config.held_out_subjects
config.folds = dual_config.folds
config.held_out_fold = True
config.nb_folds = 1

###########################################
###########################################
load_segments = True                                          # Boolean to load generators from file
save_segments = True                                         # Boolean to save the training and validation generator objects. The training generator is saved with the dataset, frame and sample type properties in the name of the file. The validation generator is always using the sequential windowed method.

main_func.train_final_model(config, results, load_segments, save_segments)

############################################
##### Multiprocessing settings for the #####
#####         predictions              #####
############################################
# import multiprocessing as mp
# mp.set_start_method('spawn')

############################################

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config, results)
