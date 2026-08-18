import os

def get_path_recording(data_path, recording):
    return os.path.join(data_path, recording[0], recording[1], f"{recording[1]}_{recording[2]}.edf")

def get_path_preprocessed_data(data_path, recording):
    if 'dtai' in data_path:
        data_path = "/cw/dtaidata/NoCsBack/2025-Epilepsy-Preprocessed"
    else:
        data_path = os.path.join(data_path, 'Preprocessed')
    return os.path.join(data_path, recording[0], recording[1], f"{recording[1]}_{recording[2]}.h5")

def get_path_tfrecord(data_path, recording, start, stop):
    if 'dtai' in data_path:
        data_path = "/cw/dtaidata/NoCsBack/2025-Epilepsy-Preprocessed/TFRecords"
    else:
        data_path = os.path.join(data_path, 'Preprocessed', 'TFRecords')
    location, subject, rec_id = recording
    file_name = f"{subject}_{rec_id}_{str(int(start*1000))}_{str(int(stop*1000))}.tfrecord"
    return os.path.join(data_path, location, subject, rec_id, file_name)

def get_path_predictions(config, name, rec, fold_i):
    return get_path_predictions_file(get_path_predictions_folder(config, name, fold_i), rec)

def get_path_predictions_file(folder_path, rec):
    """Name of a recording's prediction file inside an arbitrary folder.

    Split out of get_path_predictions so that predictions written somewhere other than the
    run's own folder -- validation predictions, see analysis/consensus_val_threshold.py --
    follow the same convention. evaluate() and get_results_rec_file() recover the recording
    by splitting this name on '__'.
    """
    return os.path.join(str(folder_path),
                        rec[0] + '__' + rec[1] + '__' + rec[2] + '__preds.h5')

def get_path_config(config, name):
    return os.path.join(config.save_dir, 'models', name, 'configs')

def get_path_results(config, name):
    return os.path.join(config.save_dir, 'results', name + '__all_results.pkl')

def get_path_model_weights(model_save_path, name):
    return os.path.join(model_save_path, 'Weights', name + '.weights.h5')

def get_path_model(config, name, fold_i):
    return os.path.join(config.save_dir, 'models', name, 'fold_{}'.format(fold_i))

def get_path_predictions_folder(config, name, fold_i):
    return os.path.join(config.save_dir, 'predictions', name, 'fold_{}'.format(fold_i))

def get_paths_segments_train(config, name, fold_i):
    return os.path.join(config.save_dir, 'segments', name, 'fold_{}'.format(fold_i), 'gen_train' + '.pkl')

def get_paths_segments_val(config, name, fold_i):
    return os.path.join(config.save_dir, 'segments', name, 'fold_{}'.format(fold_i), 'gen_val' + '.pkl')