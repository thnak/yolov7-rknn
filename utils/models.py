import cv2
from rknn.api import RKNN
import numpy as np
from pathlib import Path
import onnxruntime as ort
import torch
from utils.general import non_max_suppression


class RKNN_model(object):
    def __init__(self, model_path: str, verbose: bool, quantization=False,
                 dataset=None, target=None, target_sub_class=None, device_id=None, core_mask=RKNN.NPU_CORE_AUTO):
        model_path = Path(model_path)
        dataset = Path(dataset)
        model_path = model_path.with_suffix('.onnx')

        providers = ort.get_available_providers()
        cpu_device = any(
            x in ['DmlExecutionProvider', 'CUDAExecutionProvider', 'TRTExecutionProvider'] for x in providers)
        session_opt = ort.SessionOptions()
        session_opt.enable_profiling = False
        session_opt.log_severity_level = 3
        session_opt.optimized_model_filepath = 'optim.onnx'
        session_opt.use_deterministic_compute = True
        session_opt.enable_mem_pattern = False if 'DmlExecutionProvider' in providers else True
        session_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL if cpu_device else ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(path_or_bytes=model_path.as_posix(), providers=providers,
                                       provider_options=session_opt)
        session.enable_fallback()

        self.model_inputs = [x.name for x in session.get_inputs()]
        self.model_outputs = [x.name for x in session.get_outputs()]
        self.input_shape = session.get_inputs()[0].shape
        self.use_reorg = True if self.input_shape[1] == 12 else False
        meta_datas = session.get_modelmeta().custom_metadata_map
        self.stride = eval(meta_datas.get('stride', '[8.0, 16.0, 32.0]'))
        self.names = eval(meta_datas.get('names', ''))
        self.nc = len(self.names)

        anchors = eval(meta_datas.get('anchors', None))
        self.nl = len(anchors)
        self.na = len(anchors[0])  # number of anchors

        self.anchors = torch.tensor(anchors)
        anchor_gid = eval(meta_datas.get("anchor_grid", None))
        self.anchor_grid = torch.tensor(anchor_gid)

        del session, providers, session_opt
        if self.use_reorg and quantization:
            with open(dataset.as_posix(), 'r') as data:
                dataset_imgs = data.read()
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            datas = ""
            dataset_imgs = dataset_imgs.split("\n")
            for i, x in enumerate(dataset_imgs):
                x = Path(x)
                if x.is_file():
                    img = cv2.imread(x.as_posix())
                    img = cv2.resize(img, (1280, 1280))
                    img_ = [img[::2, ::2, :], img[1::2, ::2, :], img[::2, 1::2, :], img[1::2, 1::2, :]]
                    img_ = np.concatenate(img_, axis=2)
                    img_ = np.transpose(img_, [2, 0, 1])
                    f = data_dir / f"feed_{i}.npy"
                    np.save(f.as_posix(), img_)
                    datas += f"{f.as_posix()}\n"
            dataset = Path('dataset2.txt')
            with open(dataset.as_posix(), "w") as f:
                f.write(datas)

        input_channels = 12 if self.use_reorg else 3
        self.model = RKNN(verbose=verbose)
        self.model.config(mean_values=[[0 for x in range(input_channels)]],
                          std_values=[[255 for x in range(input_channels)]],
                          compress_weight=False,
                          model_pruning=True)
        if target is None:
            if self.model.load_onnx(model=model_path.as_posix(), inputs=self.model_inputs,
                                    input_size_list=[self.input_shape], outputs=self.model_outputs) == 0:

                self.model.build(do_quantization=quantization, dataset=dataset.as_posix(), rknn_batch_size=None)
                self.model.export_rknn(model_path.with_suffix('.rknn').as_posix())
            else:
                raise f'error when loading onnx model'
        else:
            model_path = model_path.with_suffix('.rknn')
            assert self.model.load_rknn(model_path.as_posix()) == 0, f'error when loading rknn model'

        self.model.init_runtime(target=target, target_sub_class=target_sub_class, device_id=device_id,
                                core_mask=core_mask)
        if self.use_reorg:
            self.input_shape[2:] = [x * 2 for x in self.input_shape[2:]]

    def __call__(self, inputs, conf_thres=0.45, iou_thes=.25):
        outputs = []
        if self.use_reorg:
            inputs = self.reorg(inputs)
        preds = self.model.inference(inputs=[inputs], data_format='nchw')
        for i, pred in enumerate(preds):
            pred = torch.tensor(pred)
            bs, _, ny, nx = pred.shape

            grid = self._make_grid(nx, ny).to(pred.device)
            pred = pred.view(bs, self.na, self.nc + 5, ny, nx)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            y = pred.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            outputs.append(y.view(bs, self.na * nx * ny, self.nc + 5))

        return non_max_suppression(torch.cat(outputs, dim=1), conf_thres=conf_thres, iou_thres=iou_thes)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(
            [torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return box, score

    @staticmethod
    def reorg(inputs: np.ndarray):
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                inputs = np.expand_dims(inputs, 0)
        if isinstance(inputs, torch.Tensor):
            if inputs.dim == 3:
                inputs = torch.unsqueeze(inputs, 0)
        return np.concatenate(
            [inputs[:, :, ::2, ::2], inputs[:, :, 1::2, ::2], inputs[:, :, ::2, 1::2], inputs[:, :, 1::2, 1::2]], 1)
