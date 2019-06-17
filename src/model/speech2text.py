import io
import os

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

from src.utils.io import load_audio_file
from src.utils.io import save_transcript


class Speech2Text():
    """
    Speech2Text: Transcribe given audio file asynchronously
    ---
    Attributes
    -----------
    client: google.cloud.speech.SpeechClient()
        Google Cloud Speech API client
    filename:
        filename of audio file (or URI on Google Cloud Storage)
    text:
        full transcript for given audio
    -------------------------------------------
    Functions
    -----------
    transcribe(): public
        transcribe given audio file asynchronously
    transcribe_gcs(): public
        transcribe given audio file Asynchronously, specified by gcs_uri
    """
    def __init__(self, partition, index, gcs=False):
        # para partition: which partition, train/dev/test
        # para index: the index of sample
        # para gcs: whether audio file is in Google Cloud Storage
        self.client = speech.SpeechClient()
        self.filename = load_audio_file(partition, index, gcs=gcs, verbose=True)
        self.text = ""
        if len(self.filename) == 1:
            print("\ntranscribing audio file %s" % self.filename)
        if not gcs:
            self.transcribe()
        else:
            self.transcribe_gcs()
        if len(self.text) != 0:
            save_transcript(partition, index, self.text)

    def transcribe(self):
        """transcribe given audio file asynchronously
        """
        # [START speech_transcribe_async]
        # [START speech_python_migration_async_request]
        with io.open(self.filename[0], 'rb') as audio_file:
            content = audio_file.read()
        # load audio into memory
        audio = types.RecognitionAudio(content=content)
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='tr-TR'
        )

        # [START speech_python_migration_async_response]
        operation = self.client.long_running_recognize(config, audio)
        # [END speech_python_migration_async_request]

        print("\nwaiting for operation to complete ...")
        response = operation.result(timeout=200)

        # each result for a consecutive portion of audio
        # iterate through them to get transcipts for entire audio
        for result in response.results:
            print(u"Transcript: {}".format(result.alternatives[0].transcript))
            print("Confidence: {}".format(result.alternatives[0].confidence))
            self.text += result.alternatives[0].transcript
        # [END speech_python_migration_async_response]
        # [END speech_transcribe_async]

    def transcribe_gcs(self):
        """transcribe given audio file Asynchronously, specified by gcs_uri
        """
        # [START speech_transcribe_async]
        # load audio into memory
        audio = types.RecognitionAudio(uri=self.filename[0])
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='tr-TR'
        )

        operation = self.client.long_running_recognize(config, audio)

        print("\nwaiting for operation to complete ...")
        response = operation.result(timeout=500)

        # each result for a consecutive portion of audio
        # iterate through them to get transcipts for entire audio
        for result in response.results:
            print(u"Transcript: {}".format(result.alternatives[0].transcript))
            print("Confidence: {}".format(result.alternatives[0].confidence))
            self.text += result.alternatives[0].transcript
        # [END speech_transcribe_async]