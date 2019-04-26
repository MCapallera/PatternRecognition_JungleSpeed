import logging, dtwalign

logger = logging.getLogger(__name__)


class DTW:
    def __init__(self, config):
        self.window_type = config.get('window_type', 'sakoechiba')
        self.window_size = config.getint('window_size', 1500)

    def exec(self, train_features, valid_features, train_name, valid_name):
        distance = None

        try:
            result = dtwalign.dtw(
                train_features
                , valid_features
                , window_type=self.window_type
                , window_size=self.window_size
                , dist_only=True
            )
            distance = result.distance
        except ValueError:
            logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
        except Exception as e:
            logger.warning('no alignment path for "{}" and "{}"'.format(train_name, valid_name))
            logger.error(e)

        return distance
