from utils.models import RKNN_model
from utils.general import scale_coords, plot_one_box, check_img_size
import numpy as np
import cv2
from pathlib import Path
from utils.datasets import LoadImages

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
dataset = LoadImages("inference", img_size=(height, width), stride=max(model.stride), auto=False)
for path, img, im0s, vid_cap, s, ratio, dwdh in dataset:
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    preds = model(inputs=img)

    s = s.get('string', '')
    s += f'{list(img.shape[2:])} '
    for i, pred in enumerate(preds):
        if len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)} "

            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                im0s = plot_one_box(xyxy, im0s, label=label, txtColor=(0, 0, 255), bboxColor=(255, 0, 0))
    print(f'detect: {s}')
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow('a', im0s)
    if cv2.waitKey(0) == 27:
        pass
