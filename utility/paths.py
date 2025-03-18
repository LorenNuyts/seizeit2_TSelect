import os


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
