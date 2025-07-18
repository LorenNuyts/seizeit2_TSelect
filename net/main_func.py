import os
import gc

import time

import h5py
import pickle

import keras
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K

from pympler import asizeof

from analysis.dataset import dataset_stats
from data.cross_validation import leave_one_person_out, multi_objective_grouped_stratified_cross_validation
from net.DL_config import Config
from net.key_generator import generate_data_keys_sequential, generate_data_keys_subsample, generate_data_keys_sequential_window
from net.generator_ds import build_tfrecord_dataset, SequentialGenerator
from net.routines import train_net, predict_net
from net.utils import get_metrics_scoring

from data.data import Data
from net.MiniRocket_LR import MiniRocketLR
from utility import get_recs_list
from utility.constants import SEED, Paths, Keys, Metrics
from utility.paths import get_path_predictions, get_path_config, get_path_model_weights, get_path_model, \
    get_path_predictions_folder, get_path_results, get_paths_segments_val, get_paths_segments_train

from TSelect.tselect.tselect.utils import init_metadata
from TSelect.tselect.tselect.channel_selectors.tselect import TSelect
from utility.stats import Results


def train(config, results, load_segments, save_segments):
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

    if config.cross_validation == Keys.leave_one_person_out:
        CV_generator = leave_one_person_out(config.data_path, included_locations=config.locations,
                                            validation_set=config.validation_percentage)
    elif config.cross_validation == Keys.stratified:
        info_per_group = dataset_stats(config.data_path, os.path.join(config.save_dir, "dataset_stats"), config.locations)
        CV_generator = multi_objective_grouped_stratified_cross_validation(info_per_group, group_column='hospital',
                                                                           id_column='subject',
                                                                           n_splits=config.n_folds,
                                                                           train_size=config.train_percentage,
                                                                           val_size=config.validation_percentage,
                                                                           weights_columns = {'n_seizures': 0.4,
                                                                                              'hours_of_data': 0.4},
                                                                           seed=SEED)
    else:
        raise NotImplementedError('Cross-validation method not implemented yet')

    for fold_i, (train_subjects, validation_subjects, test_subject) in enumerate(CV_generator):
        K.clear_session()
        gc.collect()
        model_save_path = get_path_model(config, name, fold_i)
        path_last_epoch_callback = os.path.join(model_save_path, 'Callbacks', name + f'_{config.nb_epochs:02d}.weights.h5')
        if os.path.exists(model_save_path) and os.path.exists(path_last_epoch_callback):
            print('    | Model of fold {} already exists'.format(fold_i))
            continue
        print('Fold {}'.format(fold_i))
        print('     | Test: {}'.format(test_subject))
        print('     | Validation: {}'.format(validation_subjects))
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
            # gen_train: SegmentedGenerator = SegmentedGenerator(config, train_recs_list, train_segments, batch_size=config.batch_size, shuffle=True)
            # gen_train = build_segment_dataset(config, train_recs_list, train_segments, batch_size=config.batch_size,
            #                                   shuffle=True)
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
            # gen_val: SequentialGenerator = SequentialGenerator(config, val_recs_list, val_segments, batch_size=600, shuffle=False)
            # gen_val = build_segment_dataset(config, val_recs_list, val_segments, batch_size=600, shuffle=False)
            gen_val, validation_steps = build_tfrecord_dataset(config, val_recs_list, val_segments, batch_size=config.val_batch_size,
                                             shuffle=False)


        else:
            raise ValueError('Unknown sample type: {}'.format(config.sample_type))

        if config.channel_selection:
            selection_start = time.process_time()

            print('### Selecting channels....')
            channel_selector = TSelect(random_state=SEED, evaluation_metric=config.channel_selection_settings['evaluation_metric'],
                                       irrelevant_selector_percentage=config.channel_selection_settings['irrelevant_selector_percentage'],
                                       filtering_threshold_corr=config.channel_selection_settings['corr_threshold'],
                                       irrelevant_selector_threshold=config.channel_selection_settings['irrelevant_selector_threshold'],)
            metadata = init_metadata()
            channel_selector.fit_generator(gen_train, gen_val, metadata=metadata)
            if config.selected_channels is None:
                channel_indices = channel_selector.selected_channels
                config.selected_channels = {fold_i: [sorted(config.included_channels)[i] for i in channel_selector.selected_channels]}
            else:
                channel_indices = channel_selector.selected_channels
                config.selected_channels[fold_i] = [sorted(config.included_channels)[i] for i in channel_selector.selected_channels]
            config.reload_CH(fold=fold_i)
            results.selection_time[fold_i] = time.process_time() - selection_start

            del gen_train
            del gen_val
            del channel_selector
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
            # model.fit(gen_train.data_segs, gen_train.labels[:, 0], gen_val.data_segs, gen_val.labels[:, 0])
            model_minirocket.fit(config, gen_train, gen_val, model_save_path)
            # train_net(config, model, gen_train, gen_val, model_save_path)

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


def predict(config):
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
        predict_per_fold(config, fold_i)


def predict_per_fold(config, fold_i):
    name = config.get_name()
    print('Fold {}'.format(fold_i))
    test_subjects = config.folds[fold_i]['test']
    K.clear_session()
    gc.collect()
    config.reload_CH(fold=fold_i)
    test_recs_list = get_recs_list(config.data_path, config.locations, test_subjects)
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
    for rec in tqdm(test_recs_list):
        if os.path.isfile(get_path_predictions(config, name, rec, fold_i)):
            print(rec[0] + ' ' + rec[1] + ' ' + rec[2] + ' exists. Skipping...')
        else:
            print('Predicting for recording: {} {} {}'.format(rec[0], rec[1], rec[2]))
            with tf.device('/cpu:0'):
                segments = generate_data_keys_sequential(config, [rec], verbose=False)

                # gen_test, _ = build_tfrecord_dataset(config, [rec], segments, batch_size=config.test_batch_size,
                #                                      shuffle=False, progress_bar=False,
                #                                      channel_indices=selected_channels_indices)

                gen_test = SequentialGenerator(config, [rec], segments, batch_size=config.test_batch_size,
                                               shuffle=False, verbose=False)

                config.reload_CH(fold_i)  # DO NOT REMOVE THIS

                if config.model.lower() == Keys.minirocketLR.lower():
                    y_pred, y_true = model.predict(gen_test)
                else:
                    y_pred, y_true = predict_net(gen_test, model_weights_path, model)

            os.makedirs(os.path.dirname(get_path_predictions(config, name, rec, fold_i)), exist_ok=True)
            with h5py.File(get_path_predictions(config, name, rec, fold_i), 'w') as f:
                f.create_dataset('y_pred', data=y_pred)
                f.create_dataset('y_true', data=y_true)
            config.reload_CH()
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

            metrics_th = get_results_rec_file(config, file, file_path, fold_i, pred_fs, thresholds)

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




def get_results_rec_file(config, file, file_path, fold_i, pred_fs, thresholds):
    with h5py.File(file_path, 'r') as f:
        y_pred = list(f['y_pred'])
        y_true = list(f['y_true'])

    metrics = {Metrics.get(m): [] for m in Metrics.all_keys()}
    rec = file.split('__')[:4]
    channels = config.selected_channels[fold_i] if config.channel_selection else config.included_channels
    rec_data = Data.loadData(config.data_path, rec, included_channels=channels)
    rec_data.apply_preprocess(config.fs, data_path=config.data_path, store_preprocessed=True, recording=rec)
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