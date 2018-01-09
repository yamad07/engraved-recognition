import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MultiBox(nn.Module):
    num_classes = 21
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512 ,256, 256, 256]
    
    def __init__(self):
        super(MultiBox, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*21, kernel_size=3, padding=1))

    def forward(self, x):
        loc_list = []
        conf_list = []
        for i, x in enumerate(x):
            loc = self.loc_layers[i](x)
            N = loc.size(0)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(N, -1, 4)
            loc_list.append(loc)

            conf = self.conf_layers[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(N, -1, 21)
            conf_list.append(conf)

        loc_preds = torch.cat(loc_list, 1)
        conf_preds = torch.cat(conf_list, 1)
        return loc_preds, conf_preds
