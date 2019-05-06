import unittest
from src.speech2text import Speech2Text


class TestSpeech2Text(unittest.TestCase):
    def test_transcribe(self):
        for i in range(1, 105):
            Speech2Text('train', i, gcs=True)


if __name__ == "__main__":
    unittest.main()