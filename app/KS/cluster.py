import logging, numpy
from itertools import combinations
from KS import service
from KS.dtw import DTW

logger = logging.getLogger(__name__)


class Cluster:
    __slots__ = ('transcription', 'name_to_features', 'estimated_cost_barrier', 'train_len', 'validate_len')

    def __init__(self, transcription):
        self.transcription = transcription
        self.name_to_features = {}
        self.estimated_cost_barrier = 0
        self.train_len = 0
        self.validate_len = 0

    def set_features_for_name(self, name, features, is_validation):
        self.name_to_features[name] = (features, is_validation)

    def train(self, dtw: DTW):
        train_features = list(self.get_train_features())
        fns = []

        for features_set in combinations(train_features, 2):
            fns.append(dtw.create_delayed(features_set[0][1], features_set[1][1], features_set[0][0], features_set[1][0]))

        if len(fns) > 0:
            result = service.get_parallel()(fns)
            result = numpy.append(list(filter(None.__ne__, result)), 0)
            self.estimated_cost_barrier = numpy.max(result)
        else:
            self.estimated_cost_barrier = 0

    def compute_stats(self):
        self.train_len = len(list(self.get_train_features()))
        self.validate_len = len(self.name_to_features) - self.train_len

    def get_train_features(self):
        for name, features_set in self.name_to_features.items():
            if not features_set[1]:
                yield name, features_set[0]

    def get_validation_features(self):
        for name, features_set in self.name_to_features.items():
            if features_set[1]:
                yield name, features_set[0]
