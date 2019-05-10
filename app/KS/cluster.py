import logging, numpy, math
from KS import service

logger = logging.getLogger(__name__)


class Cluster:
    __slots__ = ('transcription', 'name_to_features', 'cost_threshold', 'train_len', 'validate_len', 'recall', 'precision')

    def __init__(self, transcription):
        self.transcription = transcription
        self.name_to_features = {}
        self.cost_threshold = 0
        self.train_len = 0
        self.validate_len = 0
        self.recall = 0
        self.precision = 0

    def set_features_for_name(self, name, features, is_validation):
        self.name_to_features[name] = (features, is_validation)

    def train(self, result_tree):
        if self.train_len == 0:
            return  # nothing to train

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

            break  # finish after first item, because all comparison are in the sub object

        self.compute_stats(same_results, differ_results)

    def compute_stats(self, same_results, differ_results):
        same_results_length = len(same_results)
        if same_results_length != self.train_len - 1:
            logger.warning('cluster "{}" has "None" in inter cluster distances ({}/{})'.format(self.transcription, same_results_length, self.train_len))

        # distance between self
        same_results.append(0)
        same_results = numpy.asarray(same_results)
        differ_results = numpy.asarray(differ_results)
        same_max = numpy.max(same_results)
        differ_min = numpy.min(differ_results)

        if same_max < differ_min:
            self.cost_threshold = same_max
        else:
            same_intersect_results = same_results[same_results > differ_min]
            differ_intersect_results = differ_results[differ_results < same_max]
            intersect_sorted = numpy.sort(numpy.concatenate((same_intersect_results, differ_intersect_results)))
            scores = [(0, 0)]
            for d in intersect_sorted:
                score = same_results[same_results < d].shape[0] - differ_intersect_results[differ_intersect_results < d].shape[0]
                scores.append((score, d))
            scores = list(reversed(scores))
            self.cost_threshold = scores[numpy.argmax(scores, axis=0)[0]][1]

        if self.cost_threshold == 0:
            self.cost_threshold = max(differ_min - 1, 1)

        selected_right = same_results[same_results < self.cost_threshold].shape[0]
        selected_all = selected_right + differ_results[differ_results < self.cost_threshold].shape[0]
        self.recall = selected_right / self.train_len
        self.precision = selected_right / selected_all if selected_all > 0 else 0

    def compute_lens(self):
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
