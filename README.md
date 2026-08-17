# README

This repository contains an example of the code to load the [SeizeIT2 dataset](https://openneuro.org/datasets/ds005873) and to train the model included in the [dataset paper](https://arxiv.org/abs/2502.01224).

# loader_test.py
Script with an example for loading files from the dataset. The classes classes.data and classes.annotation are used to create a data object, containing the signal data and extra information,  and an annotation object, containing all information regarding the seizure events of the recording.

# main_net.py
Script to train and evaluate the ChronoNet model with all parameters as in the paper. This is a suggestion of a framework that uses the data loaders and a Keras implementation of the training and evaluation routines. The data generators are likely to take a long time to run (arround 3 hours), hence the option to save the training and validation generators and load them in future runs.

## Debug runs on a handful of subjects

Every run uses the whole dataset unless you ask for a debug run: a run on a handful of subjects that
exercises the pipeline end to end within minutes. A debug run writes to its own directories
(`_debug` in the name of the run), so it can never overwrite a real model, result or prediction, and
the subjects it used are stored in its config.

Per run:
```
python main_net.py --debug        # a handful of subjects
python main_net.py --no-debug     # the whole dataset
```

To make it the default on a machine, copy `local_debug.example.json` to `local_debug.json` in the
root of the repository (git-ignored, so it stays on that machine) and adjust the subjects if you
like. `SEIZEIT2_DEBUG=0` forces a full run and `SEIZEIT2_DEBUG=1` a debug run, whatever that file
says. See `utility/debug_settings.py`.

## Conda environment setup
The python packages (and corresponding versions) used in the development of the scripts in this repository are gathered in 'environment.yml'. To easily create a conda environment with the same package versions to run the code, follow the instructions below:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create -n ENV_NAME -f environment.yml
```
