import numpy
from skimage.filters import threshold_yen


def get_background_color(img):
    threshold = threshold_yen(img)
    return numpy.mean(img[img > threshold])
