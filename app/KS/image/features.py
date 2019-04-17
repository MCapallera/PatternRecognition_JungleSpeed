from skimage.feature import hog
from skimage.io import imread

from KS.job.io.input import InputDir
from KS.job.io.output import OutputData
from KS.job.job import Job
from config import get_config_for, ConfigContainer


class ImageFeatures:
    def __init__(self, config: ConfigContainer):
        self.config = config

    def extract(self, img):
        return hog(img
                   , orientations=self.config.getint('orientations', 9)
                   , pixels_per_cell=self.config.getlist_int('pixels_per_cell', '20,20')
                   , cells_per_block=self.config.getlist_int('cells_per_block', '2,2')
                   , feature_vector=True
                   , visualize=False
                   , block_norm="L2-Hys")


class ImageFeaturesJob(Job):
    def __init__(self, name: str):
        super().__init__(name)
        config = get_config_for('job_' + name)
        self.params = config.as_dict()
        self.params['job_name'] = name
        self.output.init(self.params)
        self.features = ImageFeatures(config)

    def run(self, data):
        params = {**self.params, **data.params}
        result = {}

        for item in self.input.get_input(params):
            name = item['filename'].split('.')[0]
            result[name] = self.features.extract(imread(item['input_path']))

        params['result'] = result
        self.output.next(params)

    def create_input(self):
        return InputDir()

    def create_output(self):
        return OutputData()

