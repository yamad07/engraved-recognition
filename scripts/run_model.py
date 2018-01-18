from PIL import Image
from PIL import ImageDraw
import argparse
import glob
import os
import sys
sys.path.append('../')
from modules.models import RecognitionModel

parser = argparse.ArgumentParser(description='server test')
parser.add_argument('--filepath', default=False,help='gpu or cpu')
parser.add_argument('--box', default=None, help='rect coordination')

args = parser.parse_args()

files = glob.glob('../data/image_test/*.*')
model = RecognitionModel()

for file_path in files:
    img = Image.open(file_path)
    draw = ImageDraw.Draw(img)
    boxes = model.run(img)
    print(file_path)
    x1 = boxes['0']['x1']
    x2 = boxes['0']['x2']
    y1 = boxes['0']['y1']
    y2 = boxes['0']['y2']

    box = [x1, y1, x2, y2]

    draw.rectangle(list(box), outline='red')
    file_name = file_path.split('/')[-1]
    img.save(os.path.join('../results/', file_name))
