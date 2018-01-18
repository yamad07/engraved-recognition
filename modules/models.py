from .encoder import DataEncoder
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from .ssd import SSD
from PIL import Image
import numpy as np

class RecognitionModel:

    def __init__(self):
        self.ssd = SSD(False)
        self.ssd.load_state_dict(torch.load('../weights/engraved.pth', map_location=lambda storage, loc: storage))
        self.ssd.eval()

    def run(self, img):
        w, h = img.size
        img_resized = img.resize((300, 300))
        transform = transforms.Compose([
                                        transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img_tensor = transform(img_resized)
        loc, conf = self.ssd(Variable(img_tensor[None,:,:,:], volatile=True))
        data_encoder = DataEncoder()
        boxes, _, _ = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
        if boxes is None:
            return None
        boxes_dict = {}
        boxes_dict['width'] = w
        boxes_dict['height'] = h 
        coord_name_list = ["x1", "y1", "x2", "y2"]
        for idx, box in enumerate(boxes):
            box[::2] *= img.width
            box[1::2] *= img.height
            box_np = box.numpy()
            box_np = box_np.astype(np.int64)
            box_list = box_np.tolist()
            box_dict = {}
            for coord_name, coord in zip(coord_name_list, box_list):
                box_dict[coord_name] = coord
            boxes_dict[str(idx)] = box_dict
        return boxes_dict
