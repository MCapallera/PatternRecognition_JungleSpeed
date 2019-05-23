from os.path import join
from joblib import Parallel
from KS.transcription import Transcription


def get_transcription_provider():
    if not hasattr(get_transcription_provider, 'instance'):
        get_transcription_provider.instance = Transcription()

    return get_transcription_provider.instance


def get_parallel():
    if not hasattr(get_parallel, 'instance'):
        get_parallel.instance = Parallel(n_jobs=-2, max_nbytes='1M', verbose=1, backend='multiprocessing')

    return get_parallel.instance


def get_test_features():
    return get_test_features.image_features_dict


def update_log_dir(path):
    get_log_path.path = path


def get_log_path(filename):
    return join(get_log_path.path, filename)


get_log_path.path = '../results/ks/'
