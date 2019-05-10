import csv, math, logging, numpy

from KS import service
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from KS.service import get_log_path
from config import get_config_for
from KS.dtw import DTW
from util.dict import create_and_set

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
        cluster_dict = self.input.get_input(params)
        transcription = service.get_transcription_provider()
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
                    fns.append(self.dtw.create_delayed(valid_features, train_features, valid_name, train_name))

        results = service.get_parallel()(fns)
        result_tree = {}
        for name_one, name_two, result in results:
            if result is None:
                continue
            create_and_set(result_tree, name_one, name_two, result)

        for task in tasks:
            if task.transcription not in cluster_dict:
                logger.warning('task "{}" has no cluster'.format(task.transcription))
                continue

            cluster = cluster_dict[task.transcription]
            train_names = [name for name, features in cluster.get_train_features()]

            for valid_name, valid_features in valid_features_set_items:
                result_sub = result_tree[valid_name] if valid_name in result_tree else {}
                min_cost = numpy.min(
                    numpy.append([result_sub[name] for name in train_names if name in result_sub], math.inf))

                if cluster.cost_threshold >= min_cost:
                    task.add_selected(valid_name, min_cost)

        with open(self.config.get('output_path', get_log_path('tasks.csv')), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for task in tasks:
                data = [task.transcription]
                for name, distance in task.selected.items():
                    data.append(name)
                    data.append(distance)
                writer.writerow(data)

        params['result'] = tasks
        self.output.next(params)
