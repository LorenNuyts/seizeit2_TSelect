import copy
import os
import gc

import time
import warnings
from collections import Counter

import h5py
import pickle

import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from pympler import asizeof

from analysis.dataset import dataset_stats
from data.cross_validation import get_CV_generator, multi_objective_grouped_stratified_cross_validation
from net.DL_config import Config
from net.key_generator import generate_data_keys_sequential, generate_data_keys_subsample, generate_data_keys_sequential_window
from net.generator_ds import build_tfrecord_dataset, SequentialGenerator, SequentialGeneratorDynamic
from net.routines import train_net, predict_net
from utility.metrics import get_metrics_scoring

from data.data import Data
from net.MiniRocket_LR import MiniRocketLR
from utility import get_recs_list
from utility.constants import SEED, Paths, Keys, Metrics
from utility.paths import get_path_predictions, get_path_config, get_path_model_weights, get_path_model, \
    get_path_predictions_folder, get_path_results, get_paths_segments_val, get_paths_segments_train

from TSelect.tselect.tselect.utils import init_metadata
from TSelect.tselect.tselect.channel_selectors.tselect import TSelect
from utility.stats import Results


def train(config, results, load_segments, save_segments, fold=None):
    """ Routine to run the model's training routine.

        Args:
            config (cls): a config object with the data input type and model parameters
            results (cls): a Results object to store the results
            load_segments (bool): boolean to load the training and validation generators from a file
            save_segments (bool): boolean to save the training and validation generators
    """

    name = config.get_name()

    if config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    elif config.model.lower() != Keys.minirocketLR.lower():
        raise ValueError('Model not recognized')

    if not os.path.exists(os.path.join(config.save_dir, 'models')):
        os.makedirs(os.path.join(config.save_dir, 'models'))

    config_path = get_path_config(config, name)
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    results_path = get_path_results(config, name)
    results_dir_path = os.path.dirname(results_path)
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    config.save_config(save_path=config_path)
    results.save_results(save_path=results_path)

    CV_generator, held_out_subjects = get_CV_generator(config)
    config.held_out_subjects = held_out_subjects

    for fold_i, (train_subjects, validation_subjects, test_subject) in enumerate(CV_generator):
        if fold is not None and fold_i != fold:
            print(f"Skipping fold {fold_i} because we only want to train fold {fold}")
            continue
        K.clear_session()
        gc.collect()
        model_save_path = get_path_model(config, name, fold_i)
        model_save_path_weights = get_path_model_weights(model_save_path, name)
        # path_last_epoch_callback = os.path.join(model_save_path, 'Callbacks', name + f'_{config.nb_epochs:02d}.weights.h5')
        if os.path.exists(model_save_path) and os.path.exists(model_save_path_weights):
            print('    | Model of fold {} already exists'.format(fold_i))
            continue
        elif config.model.lower() == Keys.minirocketLR.lower() and os.path.exists(os.path.join(model_save_path, 'MiniRocketLR_model.joblib')):
            print('    | Model of fold {} already exists'.format(fold_i))
            continue
        # if fold_i in config.channel_selector.keys():
        #     print('    | Fold {} already has a channel selector'.format(fold_i))
        #     continue
        print('Fold {}'.format(fold_i))
        print('     | Test: {}'.format(test_subject))
        print('     | Validation: {}'.format(validation_subjects))
        if fold_i in config.folds.keys():
            assert set(config.folds[fold_i]['train']) == set(train_subjects), \
                f"Train subjects in fold {fold_i} do not match the expected subjects. Expected: {config.folds[fold_i]['train']}, got: {train_subjects}"
            assert set(config.folds[fold_i]['validation']) == set(validation_subjects), \
                f"Validation subjects in fold {fold_i} do not match the expected subjects. Expected: {config.folds[fold_i]['validation']}, got: {validation_subjects}"
            assert set(config.folds[fold_i]['test']) == set(test_subject), \
                f"Test subject in fold {fold_i} does not match the expected subject. Expected: {config.folds[fold_i]['test']}, got: {test_subject}"
        config.folds[fold_i] = {'train': train_subjects, 'validation': validation_subjects, 'test': test_subject}
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if config.sample_type == 'subsample':
            train_segments = None
            val_segments = None
            if load_segments:
                print('Loading segments...')
                path_segments_train = get_paths_segments_train(config, config.get_name(), fold_i)
                if os.path.exists(path_segments_train):
                    with open(path_segments_train, 'rb') as inp:
                        train_segments = pickle.load(inp)

                path_segments_val = get_paths_segments_val(config, config.get_name(), fold_i)
                if os.path.exists(path_segments_val):
                    with open(get_paths_segments_val(config, config.get_name(), fold_i), 'rb') as inp:
                        val_segments = pickle.load(inp)

            train_recs_list = get_recs_list(config.data_path, config.locations, train_subjects)

            if train_segments is None:
                train_segments = generate_data_keys_subsample(config, train_recs_list)
                if save_segments:
                    path_segments_train = get_paths_segments_train(config, config.get_name(), fold_i)
                    if not os.path.exists(os.path.dirname(path_segments_train)):
                        os.makedirs(os.path.dirname(path_segments_train))

                    with open(path_segments_train, 'wb') as outp:
                        # noinspection PyTypeChecker
                        pickle.dump(train_segments, outp, pickle.HIGHEST_PROTOCOL)

            print('Generating training dataset...')
            gen_train, steps_per_epoch = build_tfrecord_dataset(config, train_recs_list, train_segments, batch_size=config.batch_size,
                                               shuffle=True)

            val_recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)

            if val_segments is None:
                val_segments = generate_data_keys_sequential_window(config, val_recs_list, config.val_batch_size)

                if save_segments:
                    path_segments_val = get_paths_segments_val(config, config.get_name(), fold_i)
                    if not os.path.exists(os.path.dirname(path_segments_val)):
                        os.makedirs(os.path.dirname(path_segments_val))

                    with open(path_segments_val, 'wb') as outp:
                        # noinspection PyTypeChecker
                        pickle.dump(val_segments, outp, pickle.HIGHEST_PROTOCOL)

            print('Generating validation dataset...')
            gen_val, validation_steps = build_tfrecord_dataset(config, val_recs_list, val_segments, batch_size=config.val_batch_size,
                                             shuffle=False)


        else:
            raise ValueError('Unknown sample type: {}'.format(config.sample_type))

        if config.channel_selection:
            selection_start = time.process_time()

            print('### Selecting channels....')
            channel_selector: TSelect = TSelect(random_state=SEED, evaluation_metric=config.channel_selection_settings['evaluation_metric'],
                                       irrelevant_selector_percentage=config.channel_selection_settings['irrelevant_selector_percentage'],
                                       filtering_threshold_corr=config.channel_selection_settings['corr_threshold'],
                                       irrelevant_selector_threshold=config.channel_selection_settings['irrelevant_selector_threshold'],)
            metadata = init_metadata()
            channel_selector.fit_generator(gen_train, gen_val, metadata=metadata)
            if config.selected_channels is None:
                channel_indices = channel_selector.selected_channels
                config.selected_channels = {fold_i: [sorted(config.included_channels)[i] for i in channel_selector.selected_channels]}
                config.channel_selector= {fold_i: channel_selector}
            else:
                channel_indices = channel_selector.selected_channels
                config.selected_channels[fold_i] = [sorted(config.included_channels)[i] for i in channel_selector.selected_channels]
                config.channel_selector[fold_i] = channel_selector
            config.reload_CH(fold=fold_i)
            results.selection_time[fold_i] = time.process_time() - selection_start

            del gen_train
            del gen_val
            gc.collect()

            # Reset the generators
            gen_train, steps_per_epoch = build_tfrecord_dataset(config, train_recs_list, train_segments,
                                                                batch_size=config.batch_size,
                                                                shuffle=True, channel_indices=channel_indices)
            gen_val, validation_steps = build_tfrecord_dataset(config, val_recs_list, val_segments,
                                                                batch_size=config.val_batch_size,
                                                                shuffle=False, channel_indices=channel_indices)


        print(f"Size train dataset: {asizeof.asizeof(gen_train) / (1024 ** 2):.2f} MB")
        print(f"Size val dataset: {asizeof.asizeof(gen_val) / (1024 ** 2):.2f} MB")

        print('### Training model....')
        if config.model.lower() == Keys.minirocketLR.lower():
            model_minirocket = MiniRocketLR()
            start_train = time.time()
            model_minirocket.fit(config, gen_train, gen_val, model_save_path)

            end_train = time.time() - start_train

        else:
            model: keras.Model = net(config)

            start_train = time.time()

            train_net(config, model, gen_train, gen_val, model_save_path, steps_per_epoch, validation_steps)

            end_train = time.time() - start_train
        print('Total train duration = ', end_train / 60)
        results.train_time[fold_i] = end_train

        config.reload_CH()
        config.save_config(save_path=config_path)
        results.config = config
        results.save_results(save_path=results_path)

#######################################################################################################################
#######################################################################################################################

def train_final_model(config, dual_config, results, fold):
    """ Train the model on all data except the held out fold. Only a single fold is trained, aimed to estimate how well
    the previously selected channels are on unseen data.

        Args:
            config (cls): a config object with the data input type and model parameters
            dual_config (Config): a config object with the data input and model parameters of the corresponding channel
            selection experiment. This is only used to determine the train, validation and test sets.
            results (cls): a Results object to store the results

    """
    name = config.get_name()

    if config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    elif config.model.lower() != Keys.minirocketLR.lower():
        raise ValueError('Model not recognized')

    if not os.path.exists(os.path.join(config.save_dir, 'models')):
        os.makedirs(os.path.join(config.save_dir, 'models'))

    config_path = get_path_config(config, name)
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    results_path = get_path_results(config, name)
    results_dir_path = os.path.dirname(results_path)
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    config.save_config(save_path=config_path)
    results.save_results(save_path=results_path)
    for fold_i in config.folds.keys():
        if fold is not None and fold_i != fold:
            print(f"Skipping fold {fold_i} because we only want to train fold {fold}")
            continue
        train_subjects = dual_config.folds[fold_i]['train']
        validation_subjects = dual_config.folds[fold_i]['validation']
        test_subjects = dual_config.folds[fold_i]['test']

        K.clear_session()
        gc.collect()
        model_save_path = get_path_model(config, name, fold_i)
        model_save_path_weights = get_path_model_weights(model_save_path, name)
        # path_last_epoch_callback = os.path.join(model_save_path, 'Callbacks', name + f'_{config.nb_epochs:02d}.weights.h5')
        if os.path.exists(model_save_path) and os.path.exists(model_save_path_weights):
            print('    | Model of fold {} already exists'.format(fold_i))
            continue
        # if fold_i in config.channel_selector.keys():
        #     print('    | Fold {} already has a channel selector'.format(fold_i))
        #     continue
        print('Fold {}'.format(fold_i))
        print('     | Validation: {}'.format(validation_subjects))
        config.folds[fold_i] = {'train': train_subjects, 'validation': validation_subjects, 'test': config.held_out_subjects,
                                'other_test': test_subjects}
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if config.sample_type == 'subsample':
            # train_segments = None
            # val_segments = None
            print('Loading segments...')
            path_segments_train = get_paths_segments_train(dual_config, dual_config.get_name(), fold_i)
            if os.path.exists(path_segments_train):
                with open(path_segments_train, 'rb') as inp:
                    train_segments = pickle.load(inp)
            else:
                raise ValueError(f"Training segments for fold {fold_i} not found at {path_segments_train}. Please run "
                                 f"the channel selection experiment first.")

            path_segments_val = get_paths_segments_val(dual_config, dual_config.get_name(), fold_i)
            if os.path.exists(path_segments_val):
                with open(path_segments_val, 'rb') as inp:
                    val_segments = pickle.load(inp)
            else:
                raise ValueError(f"Validation segments for fold {fold_i} not found at {path_segments_val}. Please run "
                                 f"the channel selection experiment first.")

            train_recs_list = get_recs_list(config.data_path, config.locations, train_subjects)

            # if train_segments is None:
            #     train_segments = generate_data_keys_subsample(config, train_recs_list)
            #     if save_segments:
            #         path_segments_train = get_paths_segments_train(config, config.get_name(), fold_i)
            #         if not os.path.exists(os.path.dirname(path_segments_train)):
            #             os.makedirs(os.path.dirname(path_segments_train))
            #
            #         with open(path_segments_train, 'wb') as outp:
            #             # noinspection PyTypeChecker
            #             pickle.dump(train_segments, outp, pickle.HIGHEST_PROTOCOL)

            print('Generating training dataset...')
            gen_train, steps_per_epoch = build_tfrecord_dataset(config, train_recs_list, train_segments, batch_size=config.batch_size,
                                               shuffle=True)

            val_recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)

            # if val_segments is None:
            #     val_segments = generate_data_keys_sequential_window(config, val_recs_list, config.val_batch_size)
            #
            #     if save_segments:
            #         path_segments_val = get_paths_segments_val(config, config.get_name(), fold_i)
            #         if not os.path.exists(os.path.dirname(path_segments_val)):
            #             os.makedirs(os.path.dirname(path_segments_val))
            #
            #         with open(path_segments_val, 'wb') as outp:
            #             # noinspection PyTypeChecker
            #             pickle.dump(val_segments, outp, pickle.HIGHEST_PROTOCOL)

            print('Generating validation dataset...')
            gen_val, validation_steps = build_tfrecord_dataset(config, val_recs_list, val_segments, batch_size=config.val_batch_size,
                                             shuffle=False)


        else:
            raise ValueError('Unknown sample type: {}'.format(config.sample_type))

        print('### Training model....')
        if config.model.lower() == Keys.minirocketLR.lower():
            model_minirocket = MiniRocketLR()
            start_train = time.time()
            # model.fit(gen_train.data_segs, gen_train.labels[:, 0], gen_val.data_segs, gen_val.labels[:, 0])
            model_minirocket.fit(config, gen_train, gen_val, model_save_path)
            # train_net(config, model, gen_train, gen_val, model_save_path)

            end_train = time.time() - start_train

        else:
            model: keras.Model = net(config)

            start_train = time.time()

            train_net(config, model, gen_train, gen_val, model_save_path, steps_per_epoch, validation_steps)

            end_train = time.time() - start_train
        results.train_time[fold_i] = end_train

        config.reload_CH()
        config.save_config(save_path=config_path)
        results.config = config
        results.save_results(save_path=results_path)

#######################################################################################################################
#######################################################################################################################


def predict(config, fold=None):
    # import concurrent.futures
    name = config.get_name()
    config_path = get_path_config(config, name)
    config.load_config(config_path=config_path, config_name=name)

    # Loading the results from barabas on my personal computer
    if 'dtai' in config.save_dir and 'dtai' not in os.path.dirname(os.path.realpath(__file__)):
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)

    if not hasattr(config, 'test_batch_size'):
        config.test_batch_size = 6*60

    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.makedirs(os.path.join(config.save_dir, 'predictions'))
    if not os.path.exists(os.path.join(config.save_dir, 'predictions', name)):
        os.makedirs(os.path.join(config.save_dir, 'predictions', name))

    # # Run predictions in parallel
    # with concurrent.futures.ProcessPoolExecutor(max_workers=config.n_folds) as executor:
    #     futures = [executor.submit(predict_per_fold, config, i) for i in range(config.n_folds)]
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         fold_index = future.result()
    #         print(f"Fold {fold_index} completed")

    for fold_i in config.folds.keys():
        if fold is not None and fold_i != fold:
            print(f"Skipping fold {fold_i} because we only want to predict fold {fold}")
            continue
        predict_per_fold(config, fold_i)


def predict_per_fold(config, fold_i):
    name = config.get_name()
    print('Fold {}'.format(fold_i))
    test_subjects = config.folds[fold_i]['test']
    K.clear_session()
    gc.collect()
    config.reload_CH(fold=fold_i)
    test_recs_list = get_recs_list(config.data_path, config.locations, test_subjects)
    # print("Recordings to predict:", test_recs_list)
    selected_channels_indices = [sorted(config.included_channels).index(ch) for ch in
                                 config.selected_channels[fold_i]] if config.channel_selection else None
    model_save_path = get_path_model(config, name, fold_i)
    model_weights_path = get_path_model_weights(model_save_path, name)
    if config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    elif config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model.lower() != Keys.minirocketLR.lower():
        raise ValueError('Model not recognized')
    if config.model.lower() == Keys.minirocketLR.lower():
        model = MiniRocketLR(model_save_path)
    else:
        model = net(config)
        K.set_image_data_format('channels_last')
        model.load_weights(model_weights_path)
    for rec in tqdm(test_recs_list):
        if os.path.isfile(get_path_predictions(config, name, rec, fold_i)):
            pass
            print(f"Fold {fold_i}: {rec[0]} {rec[1]} {rec[2]} exists. Skipping...")
        else:
            # print('Predicting for recording: {} {} {}'.format(rec[0], rec[1], rec[2]))
            # with tf.device('/cpu:0'):
            segments = generate_data_keys_sequential(config, [rec], verbose=False)
            # print("Segments to predict:", segments)
            # print("Number of segments to predict:", len(segments))

            # gen_test, _ = build_tfrecord_dataset(config, [rec], segments, batch_size=config.test_batch_size,
            #                                      shuffle=False, progress_bar=False,
            #                                      channel_indices=selected_channels_indices)
            # gen_test.repeat(2)

            gen_test = SequentialGenerator(config, [rec], segments, batch_size=len(segments),
                                           channels=config.selected_channels[fold_i] if config.channel_selection else None,
                                           shuffle=False, verbose=False)
            # gen_test = SequentialGeneratorDynamic(config, [rec], segments, batch_size=config.test_batch_size,
            #                                channels=config.selected_channels[
            #                                    fold_i] if config.channel_selection else None,
            #                                shuffle=False, verbose=False)
            # print("Size test dataset: {:.2f} MB".format(asizeof.asizeof(gen_test) / (1024 ** 2)))

            config.reload_CH(fold_i)  # DO NOT REMOVE THIS

            # print('### Predicting model....')
            if config.model.lower() == Keys.minirocketLR.lower():
                y_pred, y_true = model.predict(gen_test)
            else:
                y_pred, y_true = predict_net(gen_test, model_weights_path, model)

            del gen_test, segments
            gc.collect()
            # K.clear_session()

            os.makedirs(os.path.dirname(get_path_predictions(config, name, rec, fold_i)), exist_ok=True)
            with h5py.File(get_path_predictions(config, name, rec, fold_i), 'w') as f:
                f.create_dataset('y_pred', data=y_pred)
                f.create_dataset('y_true', data=y_true)
            config.reload_CH()

            del y_pred, y_true
            gc.collect()

    del model
    K.clear_session()
    gc.collect()
    return fold_i


#######################################################################################################################
#######################################################################################################################


def evaluate(config: Config, results: Results):

    name = config.get_name()
    # config_path = get_path_config(config, name)
    # config.load_config(config_path, name)

    # Loading the results from barabas on my personal computer
    if 'dtai' in results.config.save_dir and 'dtai' not in os.path.dirname(os.path.realpath(__file__)):
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
        results.config.save_dir = results.config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        results.config.data_path = results.config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
    pred_fs = 1

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.makedirs(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '.h5')

    metrics = {Metrics.get(m): [] for m in Metrics.all_keys()}
    sens_ovlp_plot = []
    prec_ovlp_plot = []

    for fold_i in config.folds.keys():
        K.clear_session()
        gc.collect()
        pred_path = get_path_predictions_folder(config, name, fold_i)

        pred_files = [x for x in os.listdir(pred_path)]
        pred_files.sort()

        metrics_fold = {m: [] for m in metrics.keys()}

        for file in tqdm(pred_files):
            file_path = os.path.join(pred_path, file)

            metrics_th = get_results_rec_file(config, file, file_path, fold_i, pred_fs, thresholds,
                                              rmsa_filtering=results.rmsa_filtering)

            for m in metrics.keys():
                metrics_fold[m].append(metrics_th[m])

        for m in metrics.keys():
            metrics[m].append(np.nanmean(metrics_fold[m], axis=0))

        # to_cut = np.argmax(fah_ovlp_th)
        to_cut = np.argmax(metrics[Metrics.fah_ovlp][-1])
        fah_ovlp_plot_rec = metrics[Metrics.fah_ovlp][-1][to_cut:]
        sens_ovlp_plot_rec = metrics[Metrics.sens_ovlp][-1][to_cut:]
        prec_ovlp_plot_rec = metrics[Metrics.prec_ovlp][-1][to_cut:]

        y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in metrics[Metrics.score]]

    print('Score: ' + "%.2f" % np.nanmean(score_05))

    with h5py.File(result_file, 'w') as f:
        for m in metrics.keys():
            f.create_dataset(m, data=metrics[m])

        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)

    results.sens_ovlp = metrics[Metrics.sens_ovlp]
    results.prec_ovlp = metrics[Metrics.prec_ovlp]
    results.fah_ovlp = metrics[Metrics.fah_ovlp]
    results.f1_ovlp = metrics[Metrics.f1_ovlp]
    results.sens_epoch = metrics[Metrics.sens_epoch]
    results.spec_epoch = metrics[Metrics.spec_epoch]
    results.prec_epoch = metrics[Metrics.prec_epoch]
    results.fah_epoch = metrics[Metrics.fah_epoch]
    results.score = metrics[Metrics.score]
    results.thresholds = thresholds

    # Save the results
    results_save_path = get_path_results(config, name)
    if not results.rmsa_filtering:
        results_save_path = results_save_path.replace('.pkl', '_noRMSA.pkl')
    results.config = config
    results.save_results(results_save_path)

    # Print some metrics
    average_nb_channels = np.mean([len(chs) for chs in config.selected_channels.values()]) if config.channel_selection else config.CH

    print(f"Average score at threshold 0.5: {'%.2f' % results.average_score_th05}")
    print(f"Average F1 at threshold 0.5: {'%.2f' % results.average_f1_ovlp_th05}")
    print(f"Average FAH at threshold 0.5: {'%.2f' % results.average_fah_ovlp_th05}")
    print(f"Average Sens at threshold 0.5: {'%.2f' % results.average_sens_ovlp_th05}")
    print(f"Average Prec at threshold 0.5: {'%.2f' % results.average_prec_ovlp_th05}")
    # print(f"Best score: {'%.2f' % results.best_average_score[0]} at threshold {'%.2f' % results.best_average_score[1]}")
    # # print(f"Best F1: {'%.2f' % results.best_[0]} at threshold {'%.2f' % results.best_average_f1_ovlp[1]}")
    # # print(f"Best FAH: {'%.2f' % results.best_average_fah_ovlp[0]} at threshold {'%.2f' % results.best_average_fah_ovlp[1]}")
    # # print(f"Best Sens: {'%.2f' % results.best_average_sens_ovlp[0]} at threshold {'%.2f' % results.best_average_sens_ovlp[1]}")
    # # print(f"Best Prec: {'%.2f' % results.best_average_prec_ovlp[0]} at threshold {'%.2f' % results.best_average_prec_ovlp[1]}")
    # print(f"F1 score at best threshold: {'%.2f' % results.average_f1_ovlp_best_threshold}")
    # print(f"FAH at best threshold: {'%.2f' % results.average_fah_ovlp_best_threshold}")
    # print(f"Sens at best threshold: {'%.2f' % results.average_sens_ovlp_best_threshold}")
    # print(f"Prec at best threshold: {'%.2f' % results.average_prec_ovlp_best_threshold}")

    print("####################################################")
    print("Average selection time: " + "%.2f" % results.average_selection_time)
    print("Total time: " + "%.2f" % results.average_total_time)
    print("Average number of channels: " + "%.2f" % average_nb_channels)


def evaluate_per_lateralization(config: Config, results: Results):
    name = config.get_name()
    # config_path = get_path_config(config, name)
    # config.load_config(config_path, name)

    # Loading the results from barabas on my personal computer
    if 'dtai' in results.config.save_dir and 'dtai' not in os.path.dirname(os.path.realpath(__file__)):
        config.save_dir = config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        config.data_path = config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
        results.config.save_dir = results.config.save_dir.replace(Paths.remote_save_dir, Paths.local_save_dir)
        results.config.data_path = results.config.data_path.replace(Paths.remote_data_path, Paths.local_data_path)
    pred_fs = 1

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.makedirs(os.path.join(config.save_dir, 'results'))

    # name_left = name + '_left'
    # name_right = name + '_right'
    # name_unknown = name + '_unknown'
    # name_no_seizures = name + '_no_seizures'
    # name_bilateral = name + '_bilateral'
    # name_mixed = name + '_mixed'
    all_lateralizations = ['left', 'right', 'unknown', 'no_seizures', 'bilateral']  # , 'mixed']

    result_files = {lat: os.path.join(config.save_dir, 'results', name + lat + '.h5') for lat in all_lateralizations}

    metrics = {lat: {Metrics.get(m): [] for m in Metrics.all_keys()} for lat in all_lateralizations}
    sens_ovlp_plots = {lat: [] for lat in all_lateralizations}
    prec_ovlp_plots = {lat: [] for lat in all_lateralizations}

    for fold_i in config.folds.keys():
        K.clear_session()
        gc.collect()
        pred_path = get_path_predictions_folder(config, name, fold_i)

        pred_files = [x for x in os.listdir(pred_path)]
        pred_files.sort()

        metrics_fold = {lat: {m: [] for m in metrics[lat].keys()} for lat in all_lateralizations}

        for file in tqdm(pred_files):
            splitted_file = file.split('__')
            hospital = splitted_file[0]
            subject = splitted_file[1]
            recording = splitted_file[2]
            subject_path = os.path.join(config.data_path, hospital, subject)
            tsv_file = [f for f in os.listdir(subject_path) if f.endswith('.tsv') and recording in f][0]
            tsv_path = os.path.join(subject_path, tsv_file)
            df = pd.read_csv(tsv_path, delimiter="\t", skiprows=4)
            lateralizations = []
            for i_row, row in df.iterrows():
                if pd.isna(row['lateralization']):
                    continue
                lateralizations.append(row['lateralization'].lower())
            print(f"Lateralizations: {lateralizations} for file {file}")
            if len(lateralizations) == 0:
                lateralization = 'no_seizures'
            else:
                lat_counts = Counter(lateralizations)
                lateralization, _ = lat_counts.most_common()[0]
            # if all(['left' in lat for lat in lateralizations]):
            #     lateralization = 'left'
            # elif all(['right' in lat for lat in lateralizations]):
            #     lateralization = 'right'
            # elif all(['unknown' in lat for lat in lateralizations]):
            #     lateralization = 'unknown'
            # else:
            #     warnings.warn("Mixed lateralization found for file {}. Assigning to 'mixed' category.".format(file))
            #     lateralization = 'mixed'

            file_path = os.path.join(pred_path, file)

            metrics_th = get_results_rec_file(config, file, file_path, fold_i, pred_fs, thresholds,
                                              rmsa_filtering=results.rmsa_filtering)

            for m in metrics[lateralization].keys():
                metrics_fold[lateralization][m].append(metrics_th[m])

        for lat in metrics.keys():
            if len(metrics_fold[lat][Metrics.fah_ovlp]) == 0:
                print(f"No recordings for lateralization {lat} in fold {fold_i}. Skipping...")
                continue
            for m in metrics[lat].keys():
                metrics[lat][m].append(np.nanmean(metrics_fold[lat][m], axis=0))

            # to_cut = np.arg`max(fah_ovlp_th)
            to_cut = np.argmax(metrics[lat][Metrics.fah_ovlp][-1])
            try:
                fah_ovlp_plot_rec = metrics[lat][Metrics.fah_ovlp][-1][to_cut:]
                sens_ovlp_plot_rec = metrics[lat][Metrics.sens_ovlp][-1][to_cut:]
                prec_ovlp_plot_rec = metrics[lat][Metrics.prec_ovlp][-1][to_cut:]
                y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
                sens_ovlp_plots[lat].append(y_plot)
                y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
                prec_ovlp_plots[lat].append(y_plot)

            except IndexError as e:
                print(f"To cut: {to_cut}")
                print(f"Fah ovlp: {metrics[lat][Metrics.fah_ovlp][-1]}")
                print(f"Lateralization: {lat}")
                print(f"Nb thresholds in lateralization: {len(metrics[lat][Metrics.fah_ovlp])}")
                raise e


    for lat in metrics.keys():
        results_lat = copy.deepcopy(results)
        score_05 = [x[25] for x in metrics[lat][Metrics.score]]

        print('Score: ' + "%.2f" % np.nanmean(score_05))

        with h5py.File(result_files[lat], 'w') as f:
            for m in metrics[lat].keys():
                f.create_dataset(m, data=metrics[lat][m])

            f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plots[lat])
            f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plots[lat])
            f.create_dataset('x_plot', data=x_plot)

        results_lat.sens_ovlp = metrics[lat][Metrics.sens_ovlp]
        results_lat.prec_ovlp = metrics[lat][Metrics.prec_ovlp]
        results_lat.fah_ovlp = metrics[lat][Metrics.fah_ovlp]
        results_lat.f1_ovlp = metrics[lat][Metrics.f1_ovlp]
        results_lat.sens_epoch = metrics[lat][Metrics.sens_epoch]
        results_lat.spec_epoch = metrics[lat][Metrics.spec_epoch]
        results_lat.prec_epoch = metrics[lat][Metrics.prec_epoch]
        results_lat.fah_epoch = metrics[lat][Metrics.fah_epoch]
        results_lat.score = metrics[lat][Metrics.score]
        results_lat.thresholds = thresholds

        # Save the results
        results_save_path = get_path_results(config, name + '_' + lat)
        if not results_lat.rmsa_filtering:
            results_save_path = results_save_path.replace('.pkl', '_noRMSA.pkl')
        results_lat.config = config
        results_lat.save_results(results_save_path)

        # Print some metrics
        average_nb_channels = np.mean([len(chs) for chs in config.selected_channels.values()]) if config.channel_selection else config.CH

        print(f"Len scores: {len(results_lat.average_score_all_thresholds)}")
        print(f"Shape scores: {len(results_lat.score), len(results_lat.score[0])}")
        print(f"Average score at threshold 0.5: {'%.2f' % results_lat.average_score_th05}")
        print(f"Average F1 at threshold 0.5: {'%.2f' % results_lat.average_f1_ovlp_th05}")
        print(f"Average FAH at threshold 0.5: {'%.2f' % results_lat.average_fah_ovlp_th05}")
        print(f"Average Sens at threshold 0.5: {'%.2f' % results_lat.average_sens_ovlp_th05}")
        print(f"Average Prec at threshold 0.5: {'%.2f' % results_lat.average_prec_ovlp_th05}")

        print("####################################################")
        print("Average selection time: " + "%.2f" % results_lat.average_selection_time)
        print("Total time: " + "%.2f" % results_lat.average_total_time)
        print("Average number of channels: " + "%.2f" % average_nb_channels)



def get_results_rec_file(config, file, file_path, fold_i, pred_fs, thresholds, rmsa_filtering=True):
    with h5py.File(file_path, 'r') as f:
        y_pred = list(f['y_pred'])
        y_true = list(f['y_true'])

    metrics = {Metrics.get(m): [] for m in Metrics.all_keys()}
    rec = file.split('__')[:4]
    channels = config.selected_channels[fold_i] if config.channel_selection else config.included_channels
    rec_data = Data.loadData(config.data_path, rec, included_channels=channels)
    rec_data.apply_preprocess(config.fs, data_path=config.data_path, store_preprocessed=True, recording=rec)
    if rmsa_filtering:
        rmsa = None
        for ch in range(len(rec_data.channels)):
            ch_data = rec_data.data[ch]
            rmsa_ch = [np.sqrt(np.mean(ch_data[start:start + 2 * config.fs] ** 2)) for start in
                       range(0, len(ch_data) - 2 * config.fs + 1, 1 * config.fs)]
            rmsa_ch = [1 if 13 < rms < 150 else 0 for rms in rmsa_ch]
            if rmsa is None:
                rmsa = rmsa_ch
            else:
                rmsa = rmsa and rmsa_ch
        if len(y_pred) != len(rmsa):
            rmsa = rmsa[:len(y_pred)]
        y_pred = np.where(np.array(rmsa) == 0, 0, y_pred)
    for th in thresholds:
        sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec = get_metrics_scoring(
            y_pred, y_true, pred_fs, th)

        metrics[Metrics.sens_ovlp].append(sens_ovlp_rec)
        metrics[Metrics.prec_ovlp].append(prec_ovlp_rec)
        metrics[Metrics.fah_ovlp].append(FA_ovlp_rec)
        metrics[Metrics.f1_ovlp].append(f1_ovlp_rec)
        metrics[Metrics.sens_epoch].append(sens_epoch_rec)
        metrics[Metrics.spec_epoch].append(spec_epoch_rec)
        metrics[Metrics.prec_epoch].append(prec_epoch_rec)
        metrics[Metrics.fah_epoch].append(FA_epoch_rec)
        metrics[Metrics.f1_epoch].append(f1_epoch_rec)
        metrics[Metrics.score].append(sens_ovlp_rec * 100 - 0.4 * FA_epoch_rec)

    return metrics