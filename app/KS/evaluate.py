import logging, numpy

from KS import service
from KS.job.io.input import InputData
from KS.job.io.output import OutputData
from KS.job.job import Job
from config import get_config_for

logger = logging.getLogger(__name__)


class Evaluate(Job):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = get_config_for('job_' + name)

    def create_input(self):
        return InputData()

    def create_output(self):
        return OutputData()
    
    def run(self, data):
        params = data.params
        results = self.input.get_input(params)
        transcription_provider = service.get_transcription_provider()
        validation_names = {}

        for transcription, names in transcription_provider.transcription_to_name.items():
            for name, is_validation in names:
                if is_validation:
                    validation_names[name] = transcription

        validation_names_length = len(validation_names)
        recalls = []
        precisions = []
        for task in results:
            same = 0
            selected_right = 0
            selected_wrong = 0

            for validation_name, transcription in validation_names.items():
                is_same = transcription == task.transcription
                same += int(is_same)

                if validation_name in task.selected:
                    if is_same:
                        selected_right += 1
                    else:
                        selected_wrong += 1

            logger.info('-------[ {}'.format(task.transcription))
            logger.info('same: {}, selected_right: {}, selected_wrong: {}'.format(same, selected_right, selected_wrong))

            recall = selected_right / same if same != 0 else 1
            logger.info('recall: {}'.format(recall))

            selected = selected_right + selected_wrong
            precision = selected_right / selected if selected != 0 else int(same == 0)
            logger.info('precision: {}'.format(precision))

            accuracy = validation_names_length - (selected_wrong + (same - selected_right)) / validation_names_length
            logger.info('accuracy: {}'.format(accuracy))

            recalls.append(recall)
            precisions.append(precision)

        logger.info('-------[ summarize')
        logger.info('mean recall: {}'.format(numpy.mean(recalls)))
        logger.info('mean precision: {}'.format(numpy.mean(precisions)))