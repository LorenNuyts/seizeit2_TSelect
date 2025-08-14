import gc
import os

import keras
import tensorflow as tf
import numpy as np
from net.utils import weighted_focal_loss, sens, spec, sens_ovlp, fah_ovlp, fah_epoch, faRate_epoch, score, decay_schedule
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import LearningRateScheduler


def get_num_workers(fraction=0.5, min_workers=1, max_workers=None):
    """
    Determine number of workers based on CPU core count.

    Args:
        fraction (float): Fraction of total cores to use (e.g., 0.5 for half).
        min_workers (int): Minimum number of workers.
        max_workers (int or None): Maximum number of workers (or None for no cap).

    Returns:
        int: Number of workers to use.
    """
    total_cores = os.cpu_count() or 1
    workers = max(min_workers, int(total_cores * fraction))
    if max_workers is not None:
        workers = min(workers, max_workers)
    return workers


def train_net(config, model: keras.Model, gen_train, gen_val, model_save_path, steps_per_epoch=None, validation_steps=None):
    """ Routine to train the model with the desired configurations.

        Args:
            config: configuration object containing all parameters
            model: Keras Model object
            gen_train: a keras data generator containing the training data
            gen_val: a keras data generator containing the validation data
            model_save_path: path to the folder to save the models' weights
            steps_per_epoch: number of steps per epoch
            validation_steps: number of validation steps
    """

    K.set_image_data_format('channels_last') 

    model.summary()

    name = config.get_name()

    optimizer = Adam(learning_rate=config.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = [weighted_focal_loss]
    auc = AUC(name = 'auc')
    metrics = ['accuracy', sens, spec,sens_ovlp, fah_ovlp, fah_epoch, faRate_epoch, score, auc]

    monitor = 'val_score'
    monitor_mode = 'max'

    early_stopping = False
    patience = 50

    if not os.path.exists(os.path.join(model_save_path, 'Callbacks')):
        os.makedirs(os.path.join(model_save_path, 'Callbacks'))

    if not os.path.exists(os.path.join(model_save_path, 'History')):
        os.makedirs(os.path.join(model_save_path, 'History'))

    if not os.path.exists(os.path.join(model_save_path, 'Weights')):
        os.makedirs(os.path.join(model_save_path, 'Weights'))


    cb_model = os.path.join(model_save_path, 'Callbacks', name + '_{epoch:02d}.weights.h5')
    csv_logger = CSVLogger(os.path.join(model_save_path, 'History', name + '.csv'), append=True)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    mc = ModelCheckpoint(cb_model,
                         monitor=monitor,
                         verbose=1,
                         save_weights_only=True,
                         save_freq='epoch',
                         save_best_only=False,
                         mode=monitor_mode)


    if early_stopping:
        es = EarlyStopping(monitor=monitor,
                           patience=patience,
                           verbose=1,
                           mode='min')

    lr_sched = LearningRateScheduler(decay_schedule)

    if early_stopping:
        callbacks_list = [mc, es, csv_logger, lr_sched]
    else:
        callbacks_list = [mc, csv_logger, lr_sched]

    # # Debug code to check the shape of the first batch
    # x_batch, y_batch = next(gen_train[0])  # get the first batch
    # print("x shape:", x_batch.shape)
    # print("y shape:", y_batch.shape)
    # print("x size:", tf.size(x_batch).numpy())
    # print("expected reshape size:", 10500)

    hist = model.fit(gen_train, validation_data=gen_val,
                     epochs=config.nb_epochs,
                     callbacks=callbacks_list,
                     shuffle=False,
                     verbose=1,
                     class_weight=config.class_weights,
                     steps_per_epoch=steps_per_epoch,
                     validation_steps=validation_steps,)

    # serialize weights to HDF5
    best_model = model
    best_model.load_weights(cb_model.format(epoch=np.argmax(hist.history['val_score'])+1))
    best_model.save_weights(os.path.join(model_save_path, 'Weights', name + ".weights.h5"))

    print("Saved model to disk")


def predict_net(generator, model_weights_path, model: keras.Model):
    """ Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        generator: a keras data generator containing the data to predict
        model_weights_path: path to the folder containing the models' weights
        model: keras model object

    Returns:
        y_pred: array with the probability of seizure occurrences (0 to 1) of each consecutive
                window of the recording.
        y_true: analogous to y_pred, the array contains the label of each segment (0 or 1)
    """

    K.set_image_data_format('channels_last')

    model.load_weights(model_weights_path)

    all_y_true = []
    all_y_pred = []

    for i, (batch_x, batch_y) in enumerate(generator):
        if batch_x.shape[0] == 0:
            print(f"Batch {i} is empty")
        pred_batch = model.predict_on_batch(batch_x)
        all_y_pred.extend(pred_batch[:, 1].astype('float32'))
        all_y_true.extend(batch_y[:, 1].astype('uint8'))
        del batch_x, batch_y, pred_batch
        gc.collect()

    # Convert lists to numpy arrays once at the end
    y_pred = np.array(all_y_pred, dtype='float32')
    y_true = np.array(all_y_true, dtype='uint8')

    return y_pred, y_true
    # y_aux = []
    # for _, y in generator:
    #     y_aux.append(y)
    # # for j in range(len(generator)):
    # #     _, y = generator[j]
    # #     y_aux.append(y)
    # true_labels = np.vstack(y_aux)
    #
    # prediction = model.predict(generator, verbose=0)
    #
    # y_pred = np.empty(len(prediction), dtype='float32')
    # for j in range(len(y_pred)):
    #     y_pred[j] = prediction[j][1]
    #
    # y_true = np.empty(len(true_labels), dtype='uint8')
    # for j in range(len(y_true)):
    #     y_true[j] = true_labels[j][1]
    #
    # return y_pred, y_true
