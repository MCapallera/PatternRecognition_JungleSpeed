import csv
import numpy as np

data_folder = '../../data/mnist-csv/'


def get_training_set():
    with open(data_folder+'mnist_train.csv', 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        matrix = np.array(dataset, dtype=int)
        samples = matrix[:, 1:]
        labels = matrix[:, 0]

    return labels, samples


def get_test_set():
    with open(data_folder+'mnist_test.csv', 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        matrix = np.array(dataset, dtype=int)
        samples = matrix[:, 1:]
        labels = matrix[:, 0]

    return labels, samples


class Data:
    def __init__(self, y: np.ndarray, x: np.ndarray):
        self.y = y
        self.x = x
