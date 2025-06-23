import os

def get_path_recording(data_path, recording):
    return os.path.join(data_path, recording[0], recording[1], f"{recording[1]}_{recording[2]}.edf")

def get_path_preprocessed_data(data_path, recording):
    if 'dtai' in data_path:
        data_path = "/cw/dtaidata/NoCsBack/2025-Epilepsy-Preprocessed"
    else:
        data_path = os.path.join(data_path, 'Preprocessed')
    return os.path.join(data_path, recording[0], recording[1], f"{recording[1]}_{recording[2]}.h5")

def get_path_predictions(config, name, rec):
    return os.path.join(config.save_dir, 'predictions', name, rec[0] + '__' + rec[1] + '__'
                        + rec[2] + '__preds.h5')

def get_path_config(config, name):
    return os.path.join(config.save_dir, 'models', name, 'configs')

def get_path_results(config, name):
    return os.path.join(config.save_dir, 'results', name + '__all_results.pkl')

def get_path_model_weights(model_save_path, name):
    return os.path.join(model_save_path, 'Weights', name + '.weights.h5')

def get_path_model(config, name, fold_i):
    return os.path.join(config.save_dir, 'models', name, 'fold_{}'.format(fold_i))

def get_path_predictions_folder(config, name):
    return os.path.join(config.save_dir, 'predictions', name)

def get_paths_generators_train(config, name, fold_i):
    return os.path.join(config.save_dir, 'generators', 'fold_{}'.format(fold_i), 'gen_train_' + name + '.pkl')

def get_paths_generators_val(config, name, fold_i):
    return os.path.join(config.save_dir, 'generators', 'fold_{}'.format(fold_i), 'gen_val_' + name + '.pkl')