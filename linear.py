import torch
from rknn.api import RKNN
from pathlib import Path
import numpy as np
import time

model_dir = Path('model.pt')
model = RKNN()
model.config()
model.load_pytorch(model_dir.as_posix(), input_size_list=[[1, 1000]])
model.build(do_quantization=False)
model.export_rknn(export_path=model_dir.with_suffix('.rknn').as_posix())
model.init_runtime()
t0 = time.perf_counter()
preds = model.inference(inputs=[np.zeros([1, 1000], dtype=np.float32)])
t1 = time.perf_counter() - t0
preds = torch.tensor(np.array(preds))
values, indi = preds.topk(k=5)
print(f'values: {values}, {indi}, {t1}')
