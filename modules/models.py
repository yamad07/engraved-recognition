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
        self.ssd.load_state_dict(torch.load('../weights/weight_299.pth', map_location=lambda storage, loc: storage))
        self.ssd.eval()

    def run(self, img):
        img_resized = img.resize((300, 300))
        print(img)
        print(img_resized.size)
        transform = transforms.Compose([
                                        transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img_tensor = transform(img_resized)
        loc, conf = self.ssd(Variable(img_tensor[None,:,:,:], volatile=True))
        data_encoder = DataEncoder()
        boxes, _, _ = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
        boxes_list = []
        print(boxes)
        for box in boxes:
            print(img.width)
            print(img.height)
            box[::2] *= img.width
            box[1::2] *= img.height
            box_np= box.numpy()
            box_np = box_np.astype(np.int64)
            print(box_np)
            boxes_list.append(box_np.tolist())
        return boxes_list
