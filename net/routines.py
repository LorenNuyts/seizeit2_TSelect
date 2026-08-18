import gc
import os

import keras
import tensorflow as tf
import numpy as np
from net.utils import decay_schedule
from utility.metrics import weighted_focal_loss, sens, spec, sens_ovlp, fah_ovlp, fah_epoch, faRate_epoch, score
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import LearningRateScheduler

from utility.paths import get_path_model_weights


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
    best_model.save_weights(get_path_model_weights(model_save_path, name))

    print("Saved model to disk")


# Segments pushed through the model at a time while predicting. predict_per_fold builds its
# generator with batch_size=len(segments), so a "batch" here is a whole recording; this
# function used to hand that straight to model.predict as batch_size=batch_x.shape[0]. For an
# 18-hour recording (63425 segments at frame 2 s / stride 1 s) ChronoNet's first inception
# concatenation is then [63425, 250, 96] float32 = 6.1 GB of activations for that one tensor,
# and prediction dies with a ResourceExhaustedError on the longest recording however large the
# GPU is. Splitting the batch bounds device memory by the chunk instead of by the length of
# the recording, and does not change the predictions: none of the models here has a
# cross-sample operation, and their BatchNormalization layers use their stored moving
# statistics at inference. The near-constant chunk shape also cuts down retracing: Keras traces
# the prediction graph once for the full-size chunk plus once per distinct trailing-chunk
# length, rather than once per distinct recording length as before.
PREDICT_BATCH_SIZE = 1024


def _split_batches(generator, batch_size):
    """ Fallback for generators that cannot hand out views of their own storage.

    SequentialGenerator.iter_batches() is preferred (see predict_net); this covers anything
    else -- SequentialGeneratorDynamic, a tf.data pipeline -- by splitting whatever batches
    it produces. Slices of a batch are views, so the split itself costs no memory, but the
    generator has already materialised the batch by this point.
    """
    for batch_x, batch_y in generator:
        for start in range(0, batch_x.shape[0], batch_size):
            yield batch_x[start:start + batch_size], batch_y[start:start + batch_size]


def predict_net(generator, model_weights_path, model: keras.Model,
                batch_size: int = PREDICT_BATCH_SIZE):
    """ Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        generator: a keras data generator containing the data to predict
        model_weights_path: path to the folder containing the models' weights
        model: keras model object
        batch_size: number of segments given to the model at a time, independent of the
                    generator's own batch size (see PREDICT_BATCH_SIZE). Lower it if a
                    ResourceExhaustedError occurs; it affects memory use, not the output.

    Returns:
        y_pred: array with the probability of seizure occurrences (0 to 1) of each consecutive
                window of the recording.
        y_true: analogous to y_pred, the array contains the label of each segment (0 or 1)
    """

    # K.set_image_data_format('channels_last')
    #
    # model.load_weights(model_weights_path)

    all_y_true = []
    all_y_pred = []

    # SequentialGenerator can hand out views straight from its own storage; going through its
    # __getitem__ instead would fancy-index -- and so copy -- a whole recording per batch.
    batches = (generator.iter_batches(batch_size) if hasattr(generator, 'iter_batches')
               else _split_batches(generator, batch_size))

    for batch_x, batch_y in batches:
        if batch_x.shape[0] == 0:
            print("Empty batch encountered, skipping.")
            continue
        pred_batch = model.predict(batch_x, batch_size=batch_x.shape[0], verbose=0)
        all_y_pred.append(pred_batch[:, 1].astype('float32'))
        all_y_true.append(batch_y[:, 1].astype('uint8'))

    # Concatenate all batches into single arrays
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_true = np.concatenate(all_y_true, axis=0)

    # Guards the chunking above. get_results_rec_file silently truncates the RMSA mask to
    # len(y_pred) rather than complaining, so a short y_pred would be scored as a shortened
    # recording and quietly produce wrong metrics instead of an error.
    if len(y_pred) != len(y_true):
        raise RuntimeError(f"Predicted {len(y_pred)} segments but the generator supplied "
                           f"{len(y_true)} labels; the predictions would be scored against "
                           f"a shortened recording.")

    return y_pred, y_true
    y_aux = []
    for _, y in generator:
        y_aux.append(y)
    # for j in range(len(generator)):
    #     _, y = generator[j]
    #     y_aux.append(y)
    true_labels = np.vstack(y_aux)

    prediction = model.predict(generator, verbose=0)

    y_pred = np.empty(len(prediction), dtype='float32')
    for j in range(len(y_pred)):
        y_pred[j] = prediction[j][1]

    y_true = np.empty(len(true_labels), dtype='uint8')
    for j in range(len(y_true)):
        y_true[j] = true_labels[j][1]

    return y_pred, y_true
