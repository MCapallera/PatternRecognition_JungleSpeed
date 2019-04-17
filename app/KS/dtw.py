import math, dtwalign, numpy, logging
from KS.cluster import Cluster
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from KS.transcription import Transcription
from config import get_config_for

logger = logging.getLogger(__name__)


class DtwTrain(Job):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = get_config_for('job_' + name)

    def create_input(self):
        return InputData()

    def create_output(self):
        return OutputData()

    def run(self, data):
        params = data.params
        transcription_provider = Transcription()
        image_features_dict = self.input.get_input(params)

        cluster_dict = {}
        for transcription, names in transcription_provider.transcription_to_name.items():
            cluster = Cluster(transcription)
            for name, is_validation in names:
                if name in image_features_dict:
                    cluster.set_features_for_name(name, image_features_dict[name], is_validation)
                else:
                    logger.warning('we computed no feature for name {}'.format(name))

            cluster.train(self.config.getint('window_size', 1500))
            cluster_dict[transcription] = cluster

        params['result'] = cluster_dict
        self.output.next(params)


class DtwValidate(Job):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = get_config_for('job_' + name)

    def create_input(self):
        return InputData()

    def create_output(self):
        return OutputData()

    def run(self, data):
        params = data.params
        transcription = Transcription()
        cluster_dict = self.input.get_input(params)

        valid_features_set = {}
        for cluster_transcription, cluster in cluster_dict.items():
            for name, features in cluster.get_validation_features():
                valid_features_set[name] = features

        for task in transcription.get_tasks():
            if task.transcription not in cluster_dict:
                logger.warning('task "{}" has no cluster'.format(task.transcription))
                continue
            cluster = cluster_dict[task.transcription]
            same = 0
            selected_wrong = 0
            selected_right = 0

            for valid_name, valid_features in valid_features_set.items():
                min_cost = math.inf

                for train_name, train_features in cluster.get_train_features():
                    try:
                        result = dtwalign.dtw(
                            train_features
                            , valid_features
                            , window_type='sakoechiba'
                            , window_size=self.config.getint('window_size', 1500)
                        )
                        min_cost = min(min_cost, numpy.sum(result.get_warping_path()))
                    except ValueError:
                        logger.warning('no path found for')

                is_same_transcription = task.transcription == transcription.get_transcription_for_name(valid_name)
                same += int(is_same_transcription)

                if cluster.estimated_cost_barrier > min_cost:
                    if is_same_transcription:
                        selected_right += 1
                    else:
                        selected_wrong += 1

            if same == 0 or selected_right == 0:
                logger.info('validate failed for {}'.format(task.transcription))
            else:
                logger.info('-------[ {}'.format(task.transcription))
                logger.info('recall: {}'.format(selected_right/(selected_right + (same - selected_right))))
                logger.info('precision: {}'.format(selected_right/(selected_right + selected_wrong)))
