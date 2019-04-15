import os

from config import get_config_for


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_data_path_for_job(params):
    return
