import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class L2Norm2d(nn.Module):
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        return self.scale * x * x.pow(2).sum(dim, keepdim=True).clamp(min=1e-12).rsqrt().expand_as(x)
