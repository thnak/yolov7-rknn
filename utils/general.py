import torch
import torchvision
import time
import numpy as np
import cv2

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device='cpu')] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[..., 0].clamp_(0, img_shape[1])  # x1
    boxes[..., 1].clamp_(0, img_shape[0])  # y1
    boxes[..., 2].clamp_(0, img_shape[1])  # x2
    boxes[..., 3].clamp_(0, img_shape[0])  # y2


def plot_one_box(x, img, txtColor=None, bboxColor=None, label=None, frameinfo=[]):
    img0 = img.copy()
    h, w = img0.shape[:2]
    line_thickness = min(h, w)
    line_thickness = line_thickness / 480
    line_thickness = max(line_thickness, 1)
    tl = int(line_thickness) or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    if len(frameinfo):
        t_sizeMaxWidth = max(max(cv2.getTextSize(frameinfo[1], 0, fontScale=tl / 3, thickness=tf)[0]),
                             max(cv2.getTextSize(frameinfo[0], 0, fontScale=tl / 3, thickness=tf)[0]))
        t_sizeMaxHeight = max(min(cv2.getTextSize(frameinfo[1], 0, fontScale=tl / 3, thickness=tf)[0]),
                              min(cv2.getTextSize(frameinfo[0], 0, fontScale=tl / 3, thickness=tf)[0]))
        img0 = cv2.rectangle(img0, (0, 0), (t_sizeMaxWidth + (t_sizeMaxHeight * 2), t_sizeMaxHeight * 4), bboxColor, -1,
                             cv2.LINE_AA)
        img0 = cv2.putText(img0, frameinfo[0], org=(t_sizeMaxHeight, int(t_sizeMaxHeight * 2)),
                           fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=tl / 3, color=txtColor, thickness=1,
                           lineType=cv2.LINE_AA)
        img0 = cv2.putText(img0, frameinfo[1], org=(t_sizeMaxHeight, int(t_sizeMaxHeight * 3)),
                           fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=tl / 3, color=txtColor, thickness=1,
                           lineType=cv2.LINE_AA)
        img0 = cv2.line(img0, (0, h - 3), (int(frameinfo[2] * w), h - 3), (255, 0, 0), 5)

    if label != None and x is not None:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        img0 = cv2.rectangle(img0, c1, c2, bboxColor, thickness=tl, lineType=cv2.LINE_AA)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        img0 = cv2.rectangle(img0, c1, c2, bboxColor, -1, cv2.LINE_AA)  # filled
        img0 = cv2.drawContours(img0, [np.array([(c1[0] + t_size[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]),
                                                 (c1[0] + t_size[0] + t_size[1] + 3, c1[1])])], 0, bboxColor, -1, 16)
        img0 = cv2.putText(img0, label, org=(c1[0], c1[1] - 2), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=tl / 3,
                           color=txtColor, thickness=tf, lineType=cv2.LINE_AA)
    return img0