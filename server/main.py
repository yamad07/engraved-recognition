from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import werkzeug
import numpy as np
import io
import json
from PIL import Image
import cv2
import sys
sys.path.append('../')
from modules.models import RecognitionModel

app = Flask(__name__)
api = Api(app)


class Recognition(Resource):
    success_message_dict = {'status': 200, 'message': 'success'}
    model = RecognitionModel()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('img', type=werkzeug.FileStorage, location='files')
        args = parser.parse_args()
        img_file_storage = args['img']
        img = self.file_storage_to_np(img_file_storage)
        coord = self.model.run(img)
        self.success_message_dict["coord"] = coord
        return jsonify(self.success_message_dict)

    def file_storage_to_np(self, img_file_storage):
        in_memory_file = io.BytesIO()
        img_file_storage.save(in_memory_file)
        img_np = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = Image.fromarray(img_np)
        return img

api.add_resource(Recognition, '/recognition')
if __name__ == '__main__':
    app.run(debug=True)
