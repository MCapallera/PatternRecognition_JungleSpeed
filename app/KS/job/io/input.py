import os
from os.path import abspath, isfile, join
from KS.job.io.output import OutputDir, OutputData


class Input:
    def __init__(self):
        pass

    def accept_output(self, output):
        if output is None:
            raise Exception('output is None')
        return False

    def get_input(self, params):
        pass


class InputDir(Input):
    def accept_output(self, output):
        if isinstance(output, OutputDir):
            return True
        return False

    def get_input(self, params):
        d = abspath(params['input_directory'])
        for filename in os.listdir(os.path.abspath(d)):
            path = join(d, filename)
            if isfile(path):
                params['input_path'] = path
                params['filename'] = filename
                if 'output_directory' in params:
                    params['output_path'] = join(params['output_directory'], filename)
                yield params


class InputData(Input):
    def accept_output(self, output):
        if isinstance(output, OutputData):
            return True
        return False

    def get_input(self, params):
        return params['data']




