import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .vgg_layers import VGG
from .multibox_layers import MultiBox
from .utils import L2Norm2d
import torchvision.models as models

class SSD(nn.Module):
    def __init__(self, use_cuda=True):
        super(SSD, self).__init__()
        vgg = models.vgg16(pretrained=True)
        if use_cuda:
            self.base = list(vgg.features.cuda().children())
        else:
            self.base = list(vgg.features.children())
        self.norm4 = L2Norm2d(20)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.multibox = MultiBox()

    def forward(self, h):
        h_for_multi_box_list = []
        for idx, layer in enumerate(self.base[:23]):
            h = layer(h)
        h_for_multi_box_list.append(self.norm4(h))

        for layer in self.base[23:30]:
            h = layer(h)

        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h_for_multi_box_list.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        h_for_multi_box_list.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        h_for_multi_box_list.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        h_for_multi_box_list.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        h_for_multi_box_list.append(h)  # conv11_2
        loc_preds, conf_preds = self.multibox(h_for_multi_box_list)
        return loc_preds, conf_preds
