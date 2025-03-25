import os
import gc

import time
import h5py
import pickle
import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_3d_numpy_to_multi_index
from tqdm import tqdm

import tensorflow as tf

from data.cross_validation import leave_one_person_out
from net.key_generator import generate_data_keys_sequential, generate_data_keys_subsample, generate_data_keys_sequential_window
from net.generator_ds import SegmentedGenerator, SequentialGenerator
from net.routines import train_net, predict_net
from net.utils import get_metrics_scoring

from data.data import Data
from utility import get_recs_list
from utility.constants import SEED
from utility.paths import get_path_predictions, get_path_config, get_path_model_weights, get_path_model, \
    get_path_predictions_folder, get_path_results

from TSelect.tselect.tselect.utils import init_metadata
from TSelect.tselect.tselect.channel_selectors.tselect import TSelect

def train(config, results, load_generators, save_generators):
    """ Routine to run the model's training routine.

        Args:
            config (cls): a config object with the data input type and model parameters
            results (cls): a results object to store the results
            load_generators (bool): boolean to load the training and validation generators from file
            save_generators (bool): boolean to save the training and validation generators
    """

    name = config.get_name()

    if config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    else:
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

    if config.cross_validation == 'leave_one_person_out':
        for fold_i, (train_subjects, validation_subjects, test_subject) in enumerate(leave_one_person_out(config.data_path, included_locations=config.locations,
                                                                 validation_set=config.validation_percentage)):
            print('Fold {}'.format(fold_i))
            print('     | Test: {}'.format(test_subject))
            print('     | Validation: {}'.format(validation_subjects))
            config.folds[fold_i] = {'train': train_subjects, 'validation': validation_subjects, 'test': test_subject}
            model_save_path = get_path_model(config, name, fold_i)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_recs_list = get_recs_list(config.data_path, config.locations, train_subjects)

            if load_generators:
                print('Loading generators...')
                name = config.dataset + '_frame-' + config.frame + '_sampletype-' + config.sample_type
                with open(os.path.join('net/generators', 'gen_train_' + name + '.pkl'), 'rb') as inp:
                    gen_train = pickle.load(inp)

                with open('net/generators/gen_val.pkl', 'rb') as inp:
                    gen_val = pickle.load(inp)

            else:
                if config.sample_type == 'subsample':
                    train_segments = generate_data_keys_subsample(config, train_recs_list)
                else:
                    raise ValueError('Unknown sample type: {}'.format(config.sample_type))

                print('Generating training segments...')
                gen_train = SegmentedGenerator(config, train_recs_list, train_segments, batch_size=config.batch_size, shuffle=True)

                if save_generators:
                    name = config.dataset + '_frame-' + config.frame + '_sampletype-' + config.sample_type
                    if not os.path.exists('net/generators'):
                        os.makedirs('net/generators')

                    with open(os.path.join('net/generators', 'gen_train_' + name + '.pkl'), 'wb') as outp:
                        # noinspection PyTypeChecker
                        pickle.dump(gen_train, outp, pickle.HIGHEST_PROTOCOL)

                val_recs_list = get_recs_list(config.data_path, config.locations, validation_subjects)

                val_segments = generate_data_keys_sequential_window(config, val_recs_list, 5*60)

                print('Generating validation segments...')
                gen_val = SequentialGenerator(config, val_recs_list, val_segments, batch_size=600, shuffle=False)

                if save_generators:
                    with open('net/generators/gen_val.pkl', 'wb') as outp:
                        # noinspection PyTypeChecker
                        pickle.dump(gen_val, outp, pickle.HIGHEST_PROTOCOL)

            if config.channel_selection:
                selection_start = time.process_time()
                print('### Selecting channels....')
                channel_selector = TSelect(random_state=SEED, evaluation_metric=config.channel_selection_evaluation_metric,
                                           irrelevant_selector_percentage=config.auc_percentage,
                                           filtering_threshold_corr=config.corr_threshold)
                # channel_selector = TSelect(random_state=SEED)
                metadata = init_metadata()
                # df = from_3d_numpy_to_multi_index(gen_train.data_segs.transpose(0, 2, 1), column_names=gen_train.channels)
                df = gen_train.data_segs
                # y = pd.Series(gen_train.labels[:, 0])
                y = gen_train.labels[:, 0]
                channel_selector.fit(df, y, X_val=gen_val.data_segs, y_val=gen_val.labels[:, 0], metadata=metadata)
                selected_channels = [gen_train.channels[i] for i in channel_selector.selected_channels]
                gen_train.change_included_channels(selected_channels)
                gen_val.change_included_channels(selected_channels)
                assert gen_train.data_segs.shape[2] == gen_val.data_segs.shape[2]
                if config.selected_channels is None:
                    config.selected_channels = {fold_i: selected_channels}
                else:
                    config.selected_channels[fold_i] = selected_channels
                config.reload_CH(fold=fold_i)
                results.selection_time[fold_i] = time.process_time() - selection_start

            print('### Training model....')

            model = net(config)

            start_train = time.time()

            train_net(config, model, gen_train, gen_val, model_save_path)

            end_train = time.time() - start_train
            print('Total train duration = ', end_train / 60)
            results.train_time[fold_i] = end_train

            config.save_config(save_path=config_path)
            results.save_results(save_path=results_path)

    elif config.cross_validation == 'leave_one_seizure_out':
        raise NotImplementedError('Cross-validation method not implemented yet')
#######################################################################################################################
#######################################################################################################################


def predict(config):

    name = config.get_name()
    config_path = get_path_config(config, name)
    config.load_config(config_path=config_path, config_name=name)

    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.makedirs(os.path.join(config.save_dir, 'predictions'))
    if not os.path.exists(os.path.join(config.save_dir, 'predictions', name)):
        os.makedirs(os.path.join(config.save_dir, 'predictions', name))

    if config.cross_validation == 'leave_one_person_out':
        for fold_i in config.folds.keys():
            config.reload_CH(fold_i)
            model_save_path = get_path_model(config, name, fold_i)
            test_subject = config.folds[fold_i]['test']
            test_recs_list = get_recs_list(config.data_path, config.locations, test_subject)

            model_weights_path = get_path_model_weights(model_save_path, name)

            # config.load_config(config_path=config_path, config_name=name+'.cfg')

            if config.model == 'DeepConvNet':
                from net.DeepConv_Net import net
            elif config.model == 'ChronoNet':
                from net.ChronoNet import net
            elif config.model == 'EEGnet':
                from net.EEGnet import net
            else:
                raise ValueError('Model not recognized')

            for rec in tqdm(test_recs_list):
                if os.path.isfile(get_path_predictions(config, name, rec)):
                    print(rec[0] + ' ' + rec[1] + ' ' + rec[2] + ' exists. Skipping...')
                else:

                    with tf.device('/cpu:0'):
                        segments = generate_data_keys_sequential(config, [rec], verbose=False)

                        gen_test = SequentialGenerator(config, [rec], segments, batch_size=len(segments), shuffle=False, verbose=False)

                        if config.channel_selection:
                            gen_test.change_included_channels(config.selected_channels[fold_i])

                        config.reload_CH(fold_i)

                        model = net(config)

                        y_pred, y_true = predict_net(gen_test, model_weights_path, model)

                    with h5py.File(get_path_predictions(config, name, rec), 'w') as f:
                        f.create_dataset('y_pred', data=y_pred)
                        f.create_dataset('y_true', data=y_true)

                    gc.collect()


#######################################################################################################################
#######################################################################################################################


def evaluate(config, results):

    name = config.get_name()
    config_path = get_path_config(config, name)
    config.load_config(config_path, name)

    pred_path = get_path_predictions_folder(config, name)
    pred_fs = 1

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.makedirs(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '.h5')

    sens_ovlp = []
    prec_ovlp = []
    fah_ovlp = []
    sens_ovlp_plot = []
    prec_ovlp_plot = []
    f1_ovlp = []

    sens_epoch = []
    spec_epoch = []
    prec_epoch = []
    fah_epoch = []
    f1_epoch = []
    rocauc = []

    score = []

    pred_files = [x for x in os.listdir(pred_path)]
    pred_files.sort()

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = list(f['y_pred'])
            y_true = list(f['y_true'])

        sens_ovlp_th = []
        prec_ovlp_th = []
        fah_ovlp_th = []
        f1_ovlp_th = []

        sens_epoch_th = []
        spec_epoch_th = []
        prec_epoch_th = []
        fah_epoch_th = []
        f1_epoch_th = []
        rocauc_th = []

        score_th = []

        rec = file.split('__')[:4]
        fold_nb = None
        for fold_i in config.folds.keys():
            if rec[1] in config.folds[fold_i]['test']:
                fold_nb = fold_i
                break

        if fold_nb is None:
            raise ValueError('Recording not found in test set')

        channels = config.selected_channels[fold_nb] if config.channel_selection else config.included_channels
        rec_data = Data.loadData(config.data_path, rec, included_channels=channels)
        rec_data.apply_preprocess(config)

        rmsa = None

        for ch in range(len(rec_data.channels)):
            ch_data = rec_data.data[ch]
            rmsa_ch = [np.sqrt(np.mean(ch_data[start:start+2*config.fs]**2)) for start in range(0, len(ch_data) - 2*config.fs + 1, 1*config.fs)]
            rmsa_ch = [1 if 13 < rms < 150 else 0 for rms in rmsa_ch]
            if rmsa is None:
                rmsa = rmsa_ch
            else:
                rmsa = rmsa and rmsa_ch
        
        if len(y_pred) != len(rmsa):
            rmsa = rmsa[:len(y_pred)]
        y_pred = np.where(np.array(rmsa) == 0, 0, y_pred)

        for th in thresholds:
            sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec, rocauc_rec = get_metrics_scoring(y_pred, y_true, pred_fs, th)

            sens_ovlp_th.append(sens_ovlp_rec)
            prec_ovlp_th.append(prec_ovlp_rec)
            fah_ovlp_th.append(FA_ovlp_rec)
            f1_ovlp_th.append(f1_ovlp_rec)
            sens_epoch_th.append(sens_epoch_rec)
            spec_epoch_th.append(spec_epoch_rec)
            prec_epoch_th.append(prec_epoch_rec)
            fah_epoch_th.append(FA_epoch_rec)
            f1_epoch_th.append(f1_epoch_rec)
            rocauc_th.append(rocauc_rec)
            score_th.append(sens_ovlp_rec*100-0.4*FA_epoch_rec)

        sens_ovlp.append(sens_ovlp_th)
        prec_ovlp.append(prec_ovlp_th)
        fah_ovlp.append(fah_ovlp_th)
        f1_ovlp.append(f1_ovlp_th)

        sens_epoch.append(sens_epoch_th)
        spec_epoch.append(spec_epoch_th)
        prec_epoch.append(prec_epoch_th)
        fah_epoch.append(fah_epoch_th)
        f1_epoch.append(f1_epoch_th)
        rocauc.append(rocauc_th)

        score.append(score_th)

        to_cut = np.argmax(fah_ovlp_th)
        fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
        sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
        prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

        y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in score]

    print('Score: ' + "%.2f" % np.nanmean(score_05))

    with h5py.File(result_file, 'w') as f:
        f.create_dataset('sens_ovlp', data=sens_ovlp)
        f.create_dataset('prec_ovlp', data=prec_ovlp)
        f.create_dataset('fah_ovlp', data=fah_ovlp)
        f.create_dataset('f1_ovlp', data=f1_ovlp)
        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)
        f.create_dataset('sens_epoch', data=sens_epoch)
        f.create_dataset('spec_epoch', data=spec_epoch)
        f.create_dataset('prec_epoch', data=prec_epoch)
        f.create_dataset('fah_epoch', data=fah_epoch)
        f.create_dataset('f1_epoch', data=f1_epoch)
        f.create_dataset('rocauc', data=rocauc)
        f.create_dataset('score', data=score)

    results.sens_ovlp = sens_ovlp
    results.prec_ovlp = prec_ovlp
    results.fah_ovlp = fah_ovlp
    results.f1_ovlp = f1_ovlp
    results.rocauc = rocauc
    results.score = score
    results.thresholds = thresholds
    results.save_results(get_path_results(config, name))
    average_nb_channels = np.mean([len(chs) for chs in config.selected_channels.values()]) if config.channel_selection else config.CH

    print(f"Best score: {'%.2f' % results.best_average_score[0]} at threshold {'%.2f' % results.best_average_score[1]}")
    # print(f"Best F1: {'%.2f' % results.best_[0]} at threshold {'%.2f' % results.best_average_f1_ovlp[1]}")
    # print(f"Best FAH: {'%.2f' % results.best_average_fah_ovlp[0]} at threshold {'%.2f' % results.best_average_fah_ovlp[1]}")
    # print(f"Best Sens: {'%.2f' % results.best_average_sens_ovlp[0]} at threshold {'%.2f' % results.best_average_sens_ovlp[1]}")
    # print(f"Best Prec: {'%.2f' % results.best_average_prec_ovlp[0]} at threshold {'%.2f' % results.best_average_prec_ovlp[1]}")
    print(f"F1 score at best threshold: {'%.2f' % results.average_f1_ovlp_best_threshold}")
    print(f"FAH at best threshold: {'%.2f' % results.average_fah_ovlp_best_threshold}")
    print(f"Sens at best threshold: {'%.2f' % results.average_sens_ovlp_best_threshold}")
    print(f"Prec at best threshold: {'%.2f' % results.average_prec_ovlp_best_threshold}")
    print(f"ROCAUC at best threshold: {'%.2f' % results.average_rocauc_best_threshold}")

    print("####################################################")
    print("Average selection time: " + "%.2f" % results.average_selection_time)
    print("Total time: " + "%.2f" % results.average_total_time)
    print("Average number of channels: " + "%.2f" % average_nb_channels)

