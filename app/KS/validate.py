import math, logging
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from KS.transcription import Transcription
from config import get_config_for
from KS.dtw import DTW

logger = logging.getLogger(__name__)


class DtwValidate(Job):
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
                    result = self.dtw.exec(train_features, valid_features, train_name, valid_name)

                    if result is not None:
                        min_cost = min(min_cost, result)

                is_same_transcription = task.transcription == transcription.get_transcription_for_name(valid_name)
                same += int(is_same_transcription)

                if cluster.estimated_cost_barrier > min_cost:
                    if is_same_transcription:
                        selected_right += 1
                    else:
                        selected_wrong += 1

            if same == 0 or selected_right == 0:
                logger.info('-------[ validate failed for {}'.format(task.transcription))
                logger.info('same: {}, selected_right: {}, selected_wrong: {}'.format(same, selected_right, selected_wrong))
            else:
                logger.info('-------[ {}'.format(task.transcription))
                logger.info('recall: {}'.format(selected_right/(selected_right + (same - selected_right))))
                logger.info('precision: {}'.format(selected_right/(selected_right + selected_wrong)))
