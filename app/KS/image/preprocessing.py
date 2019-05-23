import os, numpy
from os.path import join
from PIL import Image, ImageChops, ImageDraw
from scipy.ndimage import affine_transform
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage import filters
from skimage.measure import moments_central

from KS.image.helpers import get_background_color
from KS.job.io.input import InputDir
from KS.job.io.output import OutputDir
from KS.job.job import FnJob, Job
from svgpathtools import svg2paths

from config import get_config_for
from util.dict import subset, dynamic_cast


def deskew(params):
    img = imread(params['input_path'])
    m = moments_central(img)
    c = [m[1, 0] / m[0, 0], m[0, 1] / m[0, 0]]  # cr(x), cc(y)

    if abs(c[1]) < 1e-2:
        if params['input_path'] != params['output_path']:
            imsave(params['output_path'], img)
        return

    alpha = c[1]
    affine = numpy.array([[1, 0], [alpha, 1]])
    ocenter = numpy.array(img.shape) / 2.0
    offset = c - numpy.dot(affine, ocenter)
    img = affine_transform(img, affine, offset=offset)
    imsave(params['output_path'], img)


def crop_white(params):
    image = Image.open(params['input_path'], 'r')
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()

    if 'crop_input_directory' in params and params['crop_input_directory'] != '':
        image = Image.open(join(os.path.abspath(params['crop_input_directory']), params['filename']), 'r')

    if bbox:
        if params['keep_height'] == '1':
            bbox = (bbox[0], 0, bbox[2], image.height)

        if params['width_offset'] != '0':
            width_offset = int(params['width_offset'])
            bbox = (max(bbox[0] - width_offset, 0), bbox[1], min(bbox[2] + width_offset, image.width), bbox[3])

        image = image.crop(bbox)
        image.save(params['output_path'])
    else:
        if params['output_path'] != params['input_path']:
            image.save(params['output_path'])


def scale(params):
    img = Image.open(params['input_path'], 'r')
    img = img.resize((int(params['scale_w']), int(params['scale_h'])), Image.LANCZOS)
    img.save(params['output_path'])


def crop(params):
    name = params['filename'].split('.')[0]
    paths, attributes = svg2paths(params['svg_dir'] + name + ".svg")
    img = numpy.asarray(Image.open(params['input_path'], 'r'))

    for word_index, path in enumerate(paths):
        coordinates = []
        for line in path:
            coordinates.append((line.start.real, line.start.imag))
            coordinates.append((line.end.real, line.end.imag))

        cropped = numpy.copy(img)
        if int(params['apply_polygon_mask']) == 1:
            mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
            ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
            cropped[numpy.asarray(mask) == 0] = get_background_color(img)

        cropped = Image.fromarray(cropped)
        min_x, max_x, min_y, max_y = path.bbox()
        cropped = cropped.crop((int(min_x), int(min_y), int(max_x), int(max_y)))
        path = join(params['output_directory'], '{}.png'.format(attributes[word_index]['id']))
        cropped.save(path)


def binarize(params):
    img = imread(params['input_path'])
    if len(img.shape) > 2:
        img = rgb2gray(img)

    method = params['method']
    if method not in ('sauvola', 'isodata', 'otsu', 'li', 'yen', 'local'):
        method = 'otsu'

    thresh_func = getattr(filters.thresholding, "threshold_{}".format(method))
    thresh = thresh_func(img, **dynamic_cast(subset(params, method + '_')))

    if params['keep_foreground'] == '1':
        img[img > thresh] = 255
        imsave(params['output_path'], img)
    else:
        imsave(params['output_path'], numpy.where(img > thresh, 255, 0))


class ImagePreProcessing(FnJob):
    functions = {'crop_white': crop_white, 'scale': scale, 'crop': crop, 'binarize': binarize, 'deskew': deskew}

    def create_input(self):
        return InputDir()

    def create_output(self):
        return OutputDir()


class NormalizeHeight(Job):
    def __init__(self, name: str):
        super().__init__(name)
        config = get_config_for('job_' + name)
        self.params = config.as_dict()
        self.params['job_name'] = name
        self.output.init(self.params)

    def create_input(self):
        return InputDir()

    def create_output(self):
        return OutputDir()

    def run(self, data):
        params = {**self.params, **data.params}
        heights = []
        images = []
        for item in self.input.get_input(params):
            image = Image.open(item['input_path'], 'r')
            heights.append(image.height)
            images.append((image, item['output_path']))

        height = numpy.mean(heights) if params['height'] == 'mean' else int(params['height'])
        # print(heights)
        # print(numpy.mean(heights))
        # print(numpy.median(heights))
        for image, output_path in images:
            image = image.resize((int(image.width*(height/image.height)), height), Image.LANCZOS)
            image.save(output_path)

        self.output.next(params)
