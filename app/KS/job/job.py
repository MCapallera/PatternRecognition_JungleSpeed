from KS.job.io.input import Input
from KS.job.io.output import Output
from config import get_config_for


class Job:
    def __init__(self, name: str):
        self.name = name
        self.input = self.create_input()
        self.output = self.create_output()

    def set_index(self, index: int):
        self.output.set_index(index)

    def create_input(self):
        return Input()

    def create_output(self):
        return Output()

    def run(self, data):
        pass


class FnJob(Job):
    functions = {}

    def __init__(self, name: str):
        super().__init__(name)
        config = get_config_for('job_' + name)
        self.function = self.functions[config.get('function', 'unknown')]
        self.params = config.as_dict()
        self.params['job_name'] = name
        self.output.init(self.params)

    def run(self, data):
        params = {**self.params, **data.params}
        for item in self.input.get_input(params):
            self.function(item)

        self.output.next(params)

