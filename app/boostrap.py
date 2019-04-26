import argparse, logging, pathlib, util.path, datetime
from logging import Formatter
from logging.handlers import TimedRotatingFileHandler
from config import get_config_for, config


class Bootstrap:
    def __init__(self, name: str):
        self.name = name
        self.config_file_path = ''
        self.setup_profile()
        self.setup_logger()

        logging.getLogger(self.name.upper()).info('load config:\n' + open(self.config_file_path).read())

    def setup_profile(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("profile", nargs='?', default='base')
        args = parser.parse_args()
        self.config_file_path = '../data/{}/profile/{}.cfg'.format(self.name, args.profile)

        if not pathlib.Path(self.config_file_path).exists():
            raise FileNotFoundError(self.config_file_path)

        config.read(self.config_file_path)
        config.set('main', 'data_dir', '../data/{}'.format(self.name))

    def setup_logger(self, log_level=logging.DEBUG):
        main_config = get_config_for('main')
        log_level = main_config.getint('log_level', log_level)

        logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)-8s %(name)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(self.name.upper())
        # logger.setLevel(log_level)

        if main_config.getint('log_to_file', 0) == 1:
            logger.propagate = False
            log_dir = '../results/{}/{}-{}/'.format(self.name, main_config.get('name', 'base'), datetime.datetime.now().strftime('%Y%m%d-%H%M'))
            file_path = log_dir + 'main.log'
            util.path.ensure_dir(log_dir)

            file_handler = TimedRotatingFileHandler(
                file_path
                , when='D'
                , encoding='utf-8'
            )

            file_handler.setFormatter(Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s', '%Y-%m-%d %H:%M:%S'))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
