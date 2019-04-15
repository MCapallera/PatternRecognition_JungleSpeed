import logging
from KS.dtw import DtwTrain, DtwValidate
from KS.image.preprocessing import ImagePreProcessing
from KS.job.dispatchdata import DispatchData
from config import get_config_for
logger = logging.getLogger(__name__)


class Jobs:
    def __init__(self):
        config = get_config_for('jobs')
        self.items = config.getlist('items', '')
        self.item_instances = []  # List[Job]()

    def create_jobs(self):
        for name in self.items:
            self.create_job(name)

    def create_job(self, name):
        config = get_config_for('job_' + name)
        job = job_registry[config.get('type', 'ImagePreProcessing')](name)  # type Job
        job.output.connect(self.dispatch)
        if len(self.item_instances) > 0 and not job.input.accept_output(self.item_instances[-1].output):
            raise Exception('instance "{}" rejects previous output "{}"'.format(name, self.item_instances[-1].output))

        self.item_instances.append(job)

    def update_index(self):
        for index, instance in enumerate(self.item_instances):
            instance.set_index(index)

    def dispatch(self, data: DispatchData):
        next_index = data.index + 1

        if len(self.item_instances) > next_index:
            logger.info('run job {}'.format(self.items[next_index]))
            self.item_instances[next_index].run(data)
        else:
            logger.info('jobs finished')

    def run(self):
        self.dispatch(DispatchData(-1))
        pass


job_registry = {'ImagePreProcessing': ImagePreProcessing, 'DtwTrain': DtwTrain, 'DtwValidate': DtwValidate}  # Dict[str, object]


