import io
import os

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

class Speech2Text():
    def __init__(self, filename):
        # instantiate a client
        self.client = speech.SpeechClient()
        self.filename = os.path.join('dataset', filename)

    def transcribe(self):
        # load audio into memory
        with io.open(self.filename, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US'
        )

        # detect speech in audio
        response = self.client.recognize(config, audio)
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].Transcript))