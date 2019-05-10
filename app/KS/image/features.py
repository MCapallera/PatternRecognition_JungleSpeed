import csv, math, numpy

from skimage.feature import hog
from skimage.io import imread
from KS.job.io.input import InputDir
from KS.job.io.output import OutputData
from KS.job.job import Job
from KS.service import get_log_path
from config import get_config_for, ConfigContainer


class ImageFeatures:
    def __init__(self, config: ConfigContainer):
        self.config = config

    def extract(self, img: numpy.ndarray):
        pixel_per_cell = self.config.getlist_int('pixels_per_cell', '20,20')
        cell_per_block = self.config.getlist_int('cells_per_block', '2,2')

        if self.config.getint('adapt_pixels_per_cell', 0) == 1:
            for i in [0, 1]:
                if pixel_per_cell[i] * cell_per_block[i] > img.shape[i]:
                    pixel_per_cell[i] = math.floor(img.shape[i] / cell_per_block[i])

        return hog(img
                   , orientations=self.config.getint('orientations', 9)
                   , pixels_per_cell=pixel_per_cell
                   , cells_per_block=cell_per_block
                   , feature_vector=True
                   , visualize=False
                   , block_norm="L2-Hys")


class ImageFeaturesJob(Job):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = get_config_for('job_' + name)
        self.params = self.config.as_dict()
        self.params['job_name'] = name
        self.output.init(self.params)
        self.features = ImageFeatures(self.config)

    def run(self, data):
        params = {**self.params, **data.params}
        result = {}

        for i, item in enumerate(self.input.get_input(params)):
            name = item['filename'].split('.')[0]
            try:
                result[name] = self.features.extract(imread(item['input_path']))
            except Exception as e:
                raise Exception('could not extract features for {}'.format(name), e)

            # if i > 100: break

        self.store_features(result)
        params['result'] = result
        self.output.next(params)

    def store_features(self, result):
        with open(self.config.get('output_path', get_log_path('features.csv')), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for name, features in result.items():
                writer.writerow([name, ' '.join([repr(num) for num in features])])

    def create_input(self):
        return InputDir()

    def create_output(self):
        return OutputData()

