from unittest import TestCase
from KS.transcription import Transcription


class TestTranscription(TestCase):
    def test_prepare_transcription(self):
        transcription_service = Transcription()
        keys = '\n'.join(transcription_service.transcription_to_name.keys())
        print(keys)
        self.assertTrue(keys.find('-s_cm') == -1)

    def test_get_task(self):
        transcription_dict = {}
        for task in Transcription().get_tasks():
            if task.transcription in transcription_dict:
                self.fail('not unique!')
            transcription_dict = task.transcription

