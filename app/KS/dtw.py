import logging, dtwalign, dtw, fastdtw
from scipy.spatial.distance import euclidean
from joblib import delayed
logger = logging.getLogger(__name__)


class DTW:
    def __init__(self, config):
        self.lib = config.get('lib', 'dtwalign')
        self.params = None
        self.delayed = self.__getattribute__('prepare_'+self.lib)(config)

    def create_delayed(self, train_features, valid_features, train_name, valid_name):
        if self.params is None:
            return self.delayed(train_features, valid_features, train_name, valid_name)
        else:
            return self.delayed(train_features, valid_features, train_name, valid_name, self.params)

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


def exec_dtw(train_features, valid_features, train_name, valid_name):
    d = None
    try:
        d, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(train_features, valid_features, dist='euclidean')
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
        logger.error(e)
    return d


def exec_fastdtw(train_features, valid_features, train_name, valid_name, params):
    d = None

    try:
        d, path = fastdtw.fastdtw(train_features, valid_features, dist=euclidean, **params)
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
        logger.error(e)
    return d


def exec_dtwalign(train_features, valid_features, train_name, valid_name, params):
    distance = None

    try:
        result = dtwalign.dtw(
            train_features
            , valid_features
            , dist_only=True
            , **params
        )
        distance = result.distance
    except ValueError:
        logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
    except Exception as e:
        logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
        logger.error(e)

    return distance
