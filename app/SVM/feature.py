from skimage.feature import hog
from util.dataset import Data
import numpy as np


def to_hog(data: Data):
    x_hog = np.empty((data.x.shape[0], 36), dtype=float)
    for i, x in enumerate(data.x):
        x_hog[i] = hog(x.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm="L2-Hys")
    return Data(data.y, x_hog)
