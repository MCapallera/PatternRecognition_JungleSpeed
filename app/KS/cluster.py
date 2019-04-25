from itertools import combinations
import dtwalign, numpy, logging

logger = logging.getLogger(__name__)


class Cluster:
    __slots__ = ('transcription', 'name_to_features', 'estimated_cost_barrier')

    def __init__(self, transcription):
        self.transcription = transcription
        self.name_to_features = {}
        self.estimated_cost_barrier = 0

    def set_features_for_name(self, name, features, is_validation):
        self.name_to_features[name] = (features, is_validation)

    def train(self, window_size):
        estimated_cost_barrier = 0
        train_features = list(self.get_train_features())

        for features_set in combinations(train_features, 2):
            try:
                result = dtwalign.dtw(
                    features_set[0][1]
                    , features_set[1][1]
                    , window_type='sakoechiba', window_size=window_size
                )
                estimated_cost_barrier = max(estimated_cost_barrier, numpy.sum(result.get_warping_path()))
                # logger.info('alignment path found for {} and {}'.format(features_set[0][0], features_set[1][0]))
            except ValueError:
                logger.error('no alignment path for {} and {}'.format(features_set[0][0], features_set[1][0]))
            except Exception as e:
                logger.error('no alignment path for {} and {}'.format(features_set[0][0], features_set[1][0]))
                logger.error(e)

        self.estimated_cost_barrier = estimated_cost_barrier

    def get_train_features(self):
        for name, features_set in self.name_to_features.items():
            if not features_set[1]:
                yield name, features_set[0]

    def get_validation_features(self):
        for name, features_set in self.name_to_features.items():
            if features_set[1]:
                yield name, features_set[0]

