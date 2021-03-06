import csv, logging
from itertools import combinations, permutations

from KS import service
from KS.cluster import Cluster
from KS.dtw import DTW
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from KS.service import get_log_path
from config import get_config_for
from util.dict import create_and_set

logger = logging.getLogger(__name__)


class DtwTrain(Job):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = get_config_for('job_' + name)
        self.dtw = DTW(self.config)

    def create_input(self):
        return InputData()

    def create_output(self):
        return OutputData()

    def run(self, data):
        params = data.params
        transcription_provider = service.get_transcription_provider()
        image_features_dict = self.input.get_input(params)

        train_features_set = {}
        cluster_dict = {}
        for transcription, names in transcription_provider.transcription_to_name.items():
            cluster = Cluster(transcription)
            for name, is_validation in names:
                if name in image_features_dict:
                    cluster.set_features_for_name(name, image_features_dict[name], is_validation)
                else:
                    logger.warning('could not find features for "{}"'.format(name))

            for name, features in cluster.get_train_features():
                train_features_set[name] = features

            cluster.compute_lens()
            cluster_dict[transcription] = cluster

        fns = []
        for comb in combinations(train_features_set.items(), 2):
            fns.append(self.dtw.create_delayed(comb[0][1], comb[1][1], comb[0][0], comb[1][0]))
        results = service.get_parallel()(fns)

        result_tree = {}
        for name_one, name_two, result in results:
            if result is None:
                continue

            for name in permutations([name_one, name_two], 2):
                create_and_set(result_tree, name[0], name[1], result)

        for cluster in cluster_dict.values():
            cluster.train(result_tree)

        self.store_clusters(cluster_dict)
        params['result'] = cluster_dict
        self.output.next(params)

    def store_clusters(self, cluster_dict):
        with open(self.config.get('output_path', get_log_path('clusters.csv')), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['transcription', 'cost_threshold', 'train', 'validate', 'recall', 'precision'])
            for cluster in cluster_dict.values():
                writer.writerow([
                    cluster.transcription
                    , cluster.cost_threshold
                    , cluster.train_len
                    , cluster.validate_len
                    , cluster.recall
                    , cluster.precision
                ])

