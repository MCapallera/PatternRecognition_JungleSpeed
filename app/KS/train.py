import csv, logging

from KS import service
from KS.cluster import Cluster
from KS.dtw import DTW
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from config import get_config_for

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

        cluster_dict = {}
        for transcription, names in transcription_provider.transcription_to_name.items():
            cluster = Cluster(transcription)
            for name, is_validation in names:
                if name in image_features_dict:
                    cluster.set_features_for_name(name, image_features_dict[name], is_validation)
                else:
                    logger.warning('could not find features for "{}"'.format(name))

            cluster.compute_stats()
            cluster.train(self.dtw)
            cluster_dict[transcription] = cluster

        self.store_clusters(cluster_dict)
        params['result'] = cluster_dict
        self.output.next(params)

    def store_clusters(self, cluster_dict):
        with open(self.config.get('output_path', '../data/ks/clusters.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['transcription', 'estimated_cost_barrier', 'train', 'validate'])
            for cluster in cluster_dict.values():
                writer.writerow([cluster.transcription, cluster.estimated_cost_barrier, cluster.train_len, cluster.validate_len])

