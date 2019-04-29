import logging, numpy, math
from KS import service

logger = logging.getLogger(__name__)


class Cluster:
    __slots__ = ('transcription', 'name_to_features', 'estimated_cost_barrier', 'train_len', 'validate_len', 'recall', 'precision')

    def __init__(self, transcription):
        self.transcription = transcription
        self.name_to_features = {}
        self.estimated_cost_barrier = 0
        self.train_len = 0
        self.validate_len = 0
        self.recall = 0
        self.precision = 0

    def set_features_for_name(self, name, features, is_validation):
        self.name_to_features[name] = (features, is_validation)

    def train(self, result_tree):
        train_features = dict(list(self.get_train_features()))

        same_results = []
        differ_results = []
        for name, features in train_features.items():
            if name not in result_tree:
                continue

            results = result_tree[name]
            service.get_transcription_provider()
            for n, result in results.items():
                if n in train_features:
                    same_results.append(result)
                else:
                    differ_results.append(result)

        self.calculate_stats(same_results, differ_results)

    def calculate_stats(self, same_results, differ_results):
        same_results = numpy.asarray(same_results)
        differ_results = numpy.asarray(differ_results)
        same_max = numpy.max(same_results)
        differ_min = numpy.min(differ_results)
        if same_max < differ_min:
            self.estimated_cost_barrier = same_max
        else:
            same_intersect_results = same_results[same_results > differ_min]
            differ_intersect_results = differ_results[differ_results < same_max]
            intersect_sorted = numpy.sort(numpy.concatenate((same_intersect_results, differ_intersect_results)))
            score = -math.inf
            for d in intersect_sorted:
                new_score = same_results[same_results < d].shape[0] - differ_intersect_results[differ_intersect_results < d].shape[0]
                if new_score >= score:
                    self.estimated_cost_barrier = d
                    score = new_score
                else:
                    break

        if self.estimated_cost_barrier == 0:
            self.estimated_cost_barrier = max(differ_min - 1, 1)

        selected_right = same_results[same_results < self.estimated_cost_barrier].shape[0]
        selected_all = selected_right + differ_results[differ_results < self.estimated_cost_barrier].shape[0]
        self.recall = selected_right / self.train_len if self.train_len > 0 else 0
        self.precision = selected_right / selected_all if selected_all > 0 else 0

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
