import logging

logger = logging.getLogger(__name__)


class Transcription:
    def __init__(self):
        self.transcription_to_name = {}
        self.name_to_transcription = {}
        f = open('../data/ks/ground-truth/transcription.txt', 'r')
        words = f.readlines()
        f.close()

        for w in words:
            transcription = w[10:].strip()
            name = w[0:9]
            is_valid = int(w[0:3]) >= 300

            self.name_to_transcription[name] = transcription

            if transcription in self.transcription_to_name:
                self.transcription_to_name[transcription].append((name, is_valid))
            else:
                self.transcription_to_name[transcription] = [(name, is_valid)]

    def get_transcription_for_name(self, name):
        if name in self.name_to_transcription:
            return self.name_to_transcription[name]
        logger.warning('could not find transcription for "{}"'.format(name))
        return None

    def get_name_for_transcription(self, transcription, valid=False):
        if transcription in self.transcription_to_name:
            for bag in self.transcription_to_name[transcription]:
                if bag[1] == valid:
                    return bag[0]
        logger.warning('could not find name for "{}" constraint by valid:{}'.format(transcription, valid))
        return None

    def get_tasks(self):
        f = open('../data/ks/task/keywords.txt', 'r')
        words = f.readlines()
        f.close()
        tasks = []

        for transcription in words:
            transcription = transcription.strip()
            tasks.append(Task(transcription, self.get_name_for_transcription(transcription, False)))

        return tasks


class Task:
    __slots__ = ('name', 'transcription')

    def __init__(self, transcription, name) -> None:
        self.transcription = transcription
        self.name = name

