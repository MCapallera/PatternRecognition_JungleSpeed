import math, logging, numpy

from KS import service
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
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
        transcription = service.get_transcription_provider()
        cluster_dict = self.input.get_input(params)

        valid_features_set = {}
        for cluster_transcription, cluster in cluster_dict.items():
            for name, features in cluster.get_validation_features():
                valid_features_set[name] = features

        fns = []
        tasks = transcription.get_tasks()
        valid_features_set_items = valid_features_set.items()
        for task in tasks:
            if task.transcription not in cluster_dict:
                continue

            cluster = cluster_dict[task.transcription]
            for valid_name, valid_features in valid_features_set_items:
                for train_name, train_features in cluster.get_train_features():
                    fns.append(self.dtw.create_delayed(train_features, valid_features, train_name, valid_name))
        result = service.get_parallel()(fns)

        result_index = 0
        for task in tasks:
            if task.transcription not in cluster_dict:
                logger.warning('task "{}" has no cluster'.format(task.transcription))
                continue

            cluster = cluster_dict[task.transcription]

            same = 0
            selected_wrong = 0
            selected_right = 0

            for valid_name, valid_features in valid_features_set_items:
                partial_result = result[result_index:result_index+cluster.train_len]
                result_index += cluster.train_len

                min_cost = math.inf
                partial_result = numpy.append(list(filter(None.__ne__, partial_result)), min_cost)
                min_cost = numpy.min(partial_result)

                is_same_transcription = task.transcription == transcription.get_transcription_for_name(valid_name)
                same += int(is_same_transcription)

                if cluster.estimated_cost_barrier > min_cost:
                    if is_same_transcription:
                        selected_right += 1
                    else:
                        selected_wrong += 1

            logger.info('-------[ {}'.format(task.transcription))
            logger.info('same: {}, selected_right: {}, selected_wrong: {}'.format(same, selected_right, selected_wrong))
            logger.info('recall: {}'.format(selected_right/same if same != 0 else 0))
            selected = selected_right + selected_wrong
            logger.info('precision: {}'.format(selected_right/selected if selected != 0 else 0))
