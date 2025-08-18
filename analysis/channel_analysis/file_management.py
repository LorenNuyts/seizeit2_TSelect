import os

from utility.constants import Paths
from utility.paths import get_path_config, get_path_results


def download_remote_configs(configs, local_base_dir=None, remote_base_dir=None, host=None):
    """
    Downloads remote config files to the local base directory.

    :param configs: List of config names to download.
    :param local_base_dir: Local directory where configs will be saved.
    :param remote_base_dir: Remote directory from which configs will be downloaded.
    :param host: Remote host to connect to. If None, uses the default host.
    """
    if remote_base_dir is None:
        remote_base_dir = Paths.remote_save_dir

    if host is None:
        host = Paths.remote_host

    for config in configs:
        original_save_dir = config.save_dir
        if local_base_dir is not None:
            config.save_dir = local_base_dir
        local_path = os.path.dirname(get_path_config(config, config.get_name()))
        config.save_dir = remote_base_dir
        remote_path = get_path_config(config, config.get_name())
        os.makedirs(local_path, exist_ok=True)
        os.system(f'scp -r {host}:{remote_path} {local_path} ')
        config.save_dir = original_save_dir


def download_remote_results(configs, local_base_dir=None, remote_base_dir=None, host=None):
    """
    Downloads remote results files to the local base directory.

    :param configs: List of config names to download results for.
    :param local_base_dir: Local directory where results will be saved.
    :param remote_base_dir: Remote directory from which results will be downloaded.
    :param host: Remote host to connect to. If None, uses the default host.
    """
    if remote_base_dir is None:
        remote_base_dir = Paths.remote_save_dir

    if local_base_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_base_dir = os.path.join(project_root, "..", Paths.local_save_dir)

    if host is None:
        host = Paths.remote_host

    for config in configs:
        original_save_dir = config.save_dir
        if local_base_dir is not None:
            config.save_dir = local_base_dir
        local_path = os.path.dirname(get_path_results(config, config.get_name()))
        config.save_dir = remote_base_dir
        remote_path = get_path_results(config, config.get_name())
        os.makedirs(local_path, exist_ok=True)
        os.system(f'scp {host}:{remote_path} {local_path} ')
        config.save_dir = original_save_dir