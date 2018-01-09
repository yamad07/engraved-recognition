import torch

class DataEncoder:

    def nms(self, bboxes, scores, threshold=0.5, mode='union'):
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1) * (y2-y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)

    def decode(self, loc, conf):
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.nonzero().squeeze(1)  # [#boxes,]
        print(max_conf)
        keep = self.nms(boxes[ids], max_conf[ids])
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]
