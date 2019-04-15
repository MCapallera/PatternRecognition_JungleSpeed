import configparser

import numpy


class ConfigContainer:
    def __init__(self, name):
        self.name = name

    def get(self, key, fallback):
        return config.get(self.name, key, fallback=fallback)

    def getint(self, key, fallback):
        return config.getint(self.name, key, fallback=fallback)

    def getfloat(self, key, fallback):
        return config.getfloat(self.name, key, fallback=fallback)

    def getlist(self, key, fallback):
        value = config.get(self.name, key, fallback=fallback)
        return list(filter(None, value.split(',')))

    def getlist_int(self, key, fallback):
        value = config.get(self.name, key, fallback=fallback)
        return numpy.asarray((filter(None, value.split(','))), dtype=int)

    def as_dict(self):
        return dict(config.items(self.name))


config = configparser.ConfigParser()


def get_config_for(name):
    return ConfigContainer(name)

