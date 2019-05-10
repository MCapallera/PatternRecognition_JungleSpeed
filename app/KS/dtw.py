import logging, dtwalign, dtw, fastdtw
from scipy.spatial.distance import euclidean
from joblib import delayed
logger = logging.getLogger(__name__)


class DTW:
    def __init__(self, config):
        self.lib = config.get('lib', 'dtwalign')
        self.params = None
        self.delayed = self.__getattribute__('prepare_'+self.lib)(config)

    def create_delayed(self, features_one, features_two, name_one, name_two):
        if self.params is None:
            return self.delayed(features_one, features_two, name_one, name_two)
        else:
            return self.delayed(features_one, features_two, name_one, name_two, self.params)

    def prepare_dtw(self, config):
        return delayed(exec_dtw)

    def prepare_fastdtw(self, config):
        self.params = {'radius': config.getint('fastdtw_radius', 10)}
        return delayed(exec_fastdtw)

    def prepare_dtwalign(self, config):
        self.params = {
            'window_type': config.get('window_type', 'sakoechiba')
            , 'window_size': config.getint('window_size', 1500)
        }
        return delayed(exec_dtwalign)


def exec_dtw(features_one, features_two, name_one, name_two):
    d = None
    try:
        d, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(features_one, features_two, dist='euclidean')
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(name_one, name_two))
        logger.error(e)
    return name_one, name_two, d


def exec_fastdtw(features_one, features_two, name_one, name_two, params):
    d = None

    try:
        d, path = fastdtw.fastdtw(features_one, features_two, dist=euclidean, **params)
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(name_one, name_two))
        logger.error(e)
    return name_one, name_two, d


def exec_dtwalign(features_one, features_two, name_one, name_two, params):
    distance = None

    try:
        result = dtwalign.dtw(
            features_one
            , features_two
            , dist_only=True
            , **params
        )
        distance = result.distance
    except ValueError:
        logger.warning('no alignment path for "{}" and "{}"'.format(name_one, name_two))
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(name_one, name_two))
        logger.error(e)

    return name_one, name_two, distance
