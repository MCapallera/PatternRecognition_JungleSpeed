from os import path
from KS.job.dispatchdata import DispatchData
from config import get_config_for
from util.path import ensure_dir


class Output:
    def __init__(self):
        self.next_dispatcher = None
        self.index = -1

    def init(self, params):
        pass

    def set_index(self, index: int):
        self.index = index

    def connect(self, dispatcher):
        self.next_dispatcher = dispatcher

    def next(self, params):
        self.next_dispatcher(DispatchData(self.index, params))


class OutputDir(Output):
    def init(self, params):
        if 'output_directory' not in params:
            params['output_directory'] = path.join(get_config_for('main').get('data_dir', '.'), 'job', params['job_name'])

        ensure_dir(path.join(params['output_directory'], 'test'))

    def next(self, params):
        self.next_dispatcher(DispatchData(self.index, {'input_directory': params['output_directory']}))


class OutputData(Output):
    def next(self, params):
        self.next_dispatcher(DispatchData(self.index, {'data': params['result']}))

