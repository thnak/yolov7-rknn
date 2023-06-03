from utils.models import RKNN_model
from utils.general import scale_coords, plot_one_box
import torch
import numpy as np
import cv2
from pathlib import Path


ONNX_MODEL = Path('yolov7-tiny.onnx')
DATASET = Path('dataset.txt')

if DATASET.exists():
    with open(DATASET.as_posix(), 'r') as data:
        data_s = data.read()
    data_s = data_s.split("\n")
else:
    infer_path = Path('inference')
    data_s = [x for x in infer_path.iterdir()]
    with open(DATASET.as_posix(), 'w') as data:
        s = ''
        for x in data_s:
            s += f'{x}\n'
        s = s[:-1]
        data.write(s)

model = RKNN_model(model_path=ONNX_MODEL.as_posix(), quantization=False, dataset=DATASET, verbose=True)
names = model.names
batch, channel, height, width = model.input_shape


for x in data_s:
    img = cv2.imread(x)
    if img is None: break
    im0 = img.copy()
    img = cv2.resize(img, (width, height))
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, 0)

    preds = model(inputs=img)

    s = ''
    for i, pred in enumerate(preds):
        if len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)} "

            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                im0 = plot_one_box(xyxy, im0, label=label, txtColor=(0, 0, 255), bboxColor=(255, 0, 0))
    print(f'detect: {s}')
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow('a', im0)
    if cv2.waitKey(0) == 27:
        pass

