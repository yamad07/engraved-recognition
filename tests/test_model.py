from PIL import Image
from PIL import ImageDraw
import argparse
import glob
import os
import sys
sys.path.append('../')
from modules.models import RecognitionModel
import unittest


class TestRecognitionModel(unittest.TestCase):

    def test_model(self):
        file_name = '../data/image_test/S__30564359.jpg'
        model = RecognitionModel()
        img = Image.open(file_name)
        draw = ImageDraw.Draw(img)
        boxes = model.run(img)
        actual = type(boxes)
        expected = dict 
        self.assertEqual(actual, expected)

if __name__=='__main__':
    unittest.main()
