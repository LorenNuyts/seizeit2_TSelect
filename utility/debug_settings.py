"""
Whether a run is a debug run: a run on a handful of subjects, meant to exercise the pipeline
without touching the full dataset.

A debug run is always opt-in. Without an explicit signal, everything runs on the full dataset, so a
fresh clone of this repository never silently trains on three subjects. The signals, in decreasing
order of precedence:

  1. the value passed by the caller, i.e. the --debug / --no-debug flag of the entry point;
  2. the SEIZEIT2_DEBUG environment variable ('0', 'false', 'no' or '' force a full run, any other
     value forces a debug run);
  3. local_debug.json in the root of this repository, which is git-ignored and therefore says
     something about the machine it lives on rather than about the code. See
     local_debug.example.json for its format; an empty file ({}) is enough to enable a debug run.

Only the entry points that *run* the pipeline resolve this (main_net.py, final_model.py,
final_model_reuse.py, analysis/create_preprocessed_dataset.py). Everything else defaults to a full
run, in particular the analysis scripts that look up saved runs by name: a debug run writes to its
own directories (see Config.get_name), so a debug setting leaking into an analysis script would
make it look for artefacts that do not exist.
"""

import json
import os
from typing import List, Optional, Sequence

DEBUG_SETTINGS_FILE = 'local_debug.json'
DEBUG_ENV_VAR = 'SEIZEIT2_DEBUG'

# Used when the settings file does not list them. All of these have seizures, have short recordings
# and are not among the subjects_Fz_reference, which a run with Fz_reference=False filters out.
DEFAULT_DEBUG_SUBJECTS = ['SUBJ-1a-358', 'SUBJ-1a-353', 'SUBJ-1a-471',  # Leuven Adult subjects
                          'SUBJ-7-331', 'SUBJ-7-379', 'SUBJ-7-376']  # Coimbra subjects
DEFAULT_N_DEBUG_SUBJECTS = 3

_FALSE_VALUES = {'', '0', 'false', 'no', 'off'}
_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# The settings file is git-ignored, but a directory sync (rsync, scp -r) copies it along with the
# code. A copy that lands on the cluster would turn every real run into a debug run, so the file is
# ignored there. --debug and SEIZEIT2_DEBUG=1 still work, since those are asked for explicitly.
_CLUSTER_MARKER = 'dtai'


def get_debug_settings_path() -> str:
    """ The path of the git-ignored file that marks this machine as a debug machine. """
    return os.path.join(_REPOSITORY_ROOT, DEBUG_SETTINGS_FILE)


def _read_settings_file() -> Optional[dict]:
    """ The contents of the settings file, or None if it does not exist. """
    path = get_debug_settings_path()
    if not os.path.exists(path):
        return None
    with open(path, 'r') as settings_file:
        contents = json.load(settings_file)
    assert isinstance(contents, dict), "{} must contain a JSON object, got a {}".format(path, type(contents).__name__)
    contents = {k: v for k, v in contents.items() if not k.startswith('_')}  # keys starting with _ are comments
    unknown = set(contents) - {'subjects', 'n_subjects'}
    assert not unknown, "Unknown settings in {}: {}. Known settings are 'subjects' and " \
                        "'n_subjects'.".format(path, sorted(unknown))
    return contents


def get_debug_settings(override: Optional[bool] = None, verbose: bool = True) -> Optional[dict]:
    """
    Resolves whether this is a debug run, following the order of precedence documented above.

    :param override: True forces a debug run, False forces a full run, None leaves the decision to
        the environment variable and the settings file.
    :param verbose: whether to announce a debug run.
    :return: {'subjects': [...], 'n_subjects': n} for a debug run, None for a full run.
    """
    settings = _read_settings_file()
    if settings is not None and _CLUSTER_MARKER in _REPOSITORY_ROOT:
        print("Ignoring {}: this copy of the repository lives under {}, so it is not the machine "
              "that file was written for. Pass --debug or set {}=1 for a debug run here.".format(
                  get_debug_settings_path(), _REPOSITORY_ROOT, DEBUG_ENV_VAR))
        settings = None

    if override is None and DEBUG_ENV_VAR in os.environ:
        override = os.environ[DEBUG_ENV_VAR].strip().lower() not in _FALSE_VALUES

    if override is False:
        return None
    if override is None and settings is None:
        return None

    settings = settings or {}
    resolved = {'subjects': list(settings.get('subjects', DEFAULT_DEBUG_SUBJECTS)),
                'n_subjects': int(settings.get('n_subjects', DEFAULT_N_DEBUG_SUBJECTS))}
    assert resolved['subjects'], "A debug run needs at least one subject in {}".format(get_debug_settings_path())
    assert resolved['n_subjects'] > 0, "n_subjects must be positive in {}".format(get_debug_settings_path())
    if verbose:
        source = ("--debug" if override else
                  DEBUG_ENV_VAR if DEBUG_ENV_VAR in os.environ else get_debug_settings_path())
        print("=" * 100)
        print("DEBUG RUN ({}): at most {} of the subjects {}.".format(
            source, resolved['n_subjects'], resolved['subjects']))
        print("Results of this run are NOT comparable with a full run and are written to their own "
              "directories ('_debug' in the name of the run).")
        print("=" * 100)
    return resolved


def debug_pool(debug_subjects: Optional[Sequence[str]], n_debug_subjects: int,
               available: Sequence[str], minimum: int = 1) -> List[str]:
    """
    The subjects a debug run is restricted to, given the subjects that are available.

    The configured subjects that are available are used. Since those are tied to specific hospitals,
    they can all be absent (e.g. when running another hospital); the first n_debug_subjects
    available subjects are used in that case.

    :param debug_subjects: the configured subjects of the debug run.
    :param n_debug_subjects: how many subjects to fall back to when the configured ones are absent.
    :param available: the subjects to choose from.
    :param minimum: the number of subjects the caller needs, e.g. one per subset of a split.
    """
    assert debug_subjects, "debug_pool is only meaningful for a debug run"
    available = list(dict.fromkeys(available))
    pool = [subject for subject in debug_subjects if subject in available]
    if len(pool) < max(minimum, 1):
        pool = available[:max(minimum, n_debug_subjects or DEFAULT_N_DEBUG_SUBJECTS)]
        print("Debug run: only {} of the configured debug subjects are available here, falling back "
              "to the first {}: {}. Configured: {}.".format(
                  len([s for s in debug_subjects if s in available]), len(pool), pool, list(debug_subjects)))
    assert len(pool) >= minimum, \
        "A debug run needs at least {} subjects, but only {} of {} are available".format(
            minimum, len(pool), list(debug_subjects))
    return pool
