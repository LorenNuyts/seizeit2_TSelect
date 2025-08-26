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
locations_ = [parse_location(l) for l in Locations.all_keys()]
locations_ = sorted(list(dict.fromkeys(locations_)))
config_stratified_ch_05 = get_channel_selection_config(base_, locations=locations_,
                                     evaluation_metric=evaluation_metrics['score'],
                                     irrelevant_selector_threshold=0.5, CV=Keys.stratified,
                                     held_out_fold=True)
config_stratified_ch_100 = get_channel_selection_config(base_, locations=locations_,
                                     evaluation_metric=evaluation_metrics['score'], CV=Keys.stratified,
                                     held_out_fold=True)
config_base = get_base_config(base_, locations=locations_, CV=Keys.stratified, held_out_fold=True)

stratified_configs = {
    (Nodes.CROSStop, "T7"): {
        config_stratified_ch_05: [1, 3, 4, 7, 8, 9],
        config_stratified_ch_100: [1, 3, 4, 5, 7, 9]
    },
    ("T7",) : {config_stratified_ch_05: [5]}
}

###########################################
### Parse parameters from command line ####
###########################################
parser = argparse.ArgumentParser()
parser.add_argument("--nodes", type=str, nargs="?", default="Cross_T7")
parser.add_argument('--reset', action='store_true')
parser.add_argument("--gpu", type=int, nargs="?", default=0)

args = parser.parse_args()

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

###########################################
## Initialize standard config parameters ##
###########################################
if args.nodes == "all":
    suffix_ = "_final_model_reuse_base"
elif args.nodes == "Cross_T7":
    suffix_ = "_final_model_reuse"
else:
    raise NotImplementedError(f'Nodes {args.nodes} not implemented.')
config = get_base_config(base_, locations_, suffix=suffix_, included_channels=args.nodes, held_out_fold=True)

###########################################
###########################################
config_path = get_path_config(config, config.get_name())
print('Config path:', config_path)
results_path = get_path_results(config, config.get_name())

if args.reset:
    print('Deleting results files...')
    if os.path.exists(results_path):
        os.remove(results_path)
    second_result = results_path.replace('__all_results.pkl', '.h5')
    if os.path.exists(second_result):
        os.remove(second_result)
    print("Deleting predictions folder...")
    predictions_path = os.path.join(config.save_dir, 'predictions', config.get_name())
    if os.path.exists(predictions_path):
        shutil.rmtree(predictions_path)

print("Path exists:", os.path.exists(config_path), config_path)
print("Reset?", args.reset)
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
    if not os.path.exists(config_path):
        config = results.config # If the config file is not there (because I didn't download it), load the config from the results file

config.held_out_fold = True
if args.nodes == 'all':
    config.nb_folds = len(set.union(*[set(x) for x in stratified_configs[(Nodes.CROSStop, "T7")].values()]))
else:
    config.nb_folds = len(set.union(*[set(x) for x in stratified_configs[tuple(config.included_channels)].values()]))

###########################################
###########################################
# If models don't exist yet, copy them
if args.nodes == 'all':
    config_models = {config_base: list(set.union(*[set(x) for x in stratified_configs[(Nodes.CROSStop, "T7")].values()]))}
else:
    config_models = stratified_configs[tuple(config.included_channels)]
included_folds = []
for config_model, folds in config_models.items():
    config_model_path = get_path_config(config_model, config_model.get_name())
    if os.path.exists(config_model_path):
        original_save_dir = config_model.save_dir
        config_model.load_config(config_model_path, config_model.get_name())
        config_model.save_dir = original_save_dir
    else:
        raise ValueError(f'Config file for model {config_model.get_name()} not found at {config_model_path}. Please run the channel selection experiment first.')
    for fold_i in folds:
        if fold_i in included_folds:
            continue
        new_model_save_path = get_path_model(config, config.get_name(), fold_i)
        new_model_weights_path = get_path_model_weights(new_model_save_path, config.get_name())
        reference_model_save_path = get_path_model(config_model, config_model.get_name(), fold_i)
        reference_model_weights_path = get_path_model_weights(reference_model_save_path, config_model.get_name())
        if not os.path.exists(new_model_weights_path):
            os.makedirs(os.path.dirname(new_model_weights_path), exist_ok=True)
            print(f'Copying model from {reference_model_weights_path} to {new_model_weights_path}')
            shutil.copyfile(reference_model_weights_path, new_model_weights_path)
            # shutil.copytree(reference_model_save_path, new_model_save_path)
        included_folds.append(fold_i)

        # Make sure the correct held-out fold is set as the test set
        config.folds[fold_i] = config_model.folds[fold_i]
        config.folds[fold_i]['test'] = config_model.held_out_subjects

os.makedirs(config_path, exist_ok=True)
config.save_config(save_path=config_path)
results.config = config
results.save_results(save_path=results_path)

print('Getting predictions on the held-out set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config, results)
