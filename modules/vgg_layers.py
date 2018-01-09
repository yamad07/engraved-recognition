import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def VGG():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.ReLU(True)]
            in_channels = x
    
    return nn.Sequential(*layers)
