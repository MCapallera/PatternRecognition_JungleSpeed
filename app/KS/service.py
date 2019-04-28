from joblib import Parallel
from KS.transcription import Transcription


def get_transcription_provider():
    if not hasattr(get_transcription_provider, 'instance'):
        get_transcription_provider.instance = Transcription()

    return get_transcription_provider.instance


def get_parallel():
    if not hasattr(get_parallel, 'instance'):
        get_parallel.instance = Parallel(n_jobs=-1, max_nbytes='1M', verbose=False, backend='multiprocessing')

    return get_parallel.instance
