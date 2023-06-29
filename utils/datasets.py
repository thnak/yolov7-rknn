import time
from threading import Thread
from urllib.parse import urlparse

import cv2
import glob
import numpy as np
import os
from pathlib import Path

from utils.general import clean_str

HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'pfm']  # acceptable image suffixes
VID_FORMATS = ['asf', 'mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv', 'gif']  # acceptable video suffixes
YOUTUBE = ('www.youtube.com', 'youtube.com', 'youtu.be', 'https://youtu.be')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleUp=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadImages:
    """Load image from media resouce"""

    def __init__(self, path, img_size=640, stride=32, auto=True, scaleFill=False, scaleUp=True, vid_stride=1):
        """_summary_

        Args:
            path (_type_): _description_. media file's path or txt file with media path for each line
            img_size (int, optional): _description_. Defaults to 640.
            stride (int, optional): _description_. Defaults to 32. Stride of YOLO network
            auto (bool, optional): _description_. Defaults to True. set False for rectangle shape
            transforms (_type_, optional): _description_. Defaults to None.
            vid_stride (int, optional): _description_. Defaults to 1.

        Raises:
            FileNotFoundError: _description_
        """
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleUp = scaleUp
        self.vid_stride = vid_stride  # video frame-rate stride
        self.fps = None

        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.mode = 'image'
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        im, ratio, dwdh = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto,
                                    scaleFill=self.scaleFill)  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, {'string': s, 'frames': ([self.frames] if self.mode == 'video' else [self.nf]),
                                         'c_frame': (
                                             [self.frame] if self.mode == 'video' else [self.count])}, ratio, dwdh

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees

    def __len__(self):
        return self.nf  # number of files

class LoadStreams:
    def __init__(self, sources='streams.txt', img_size=(640, 640), stride=32, auto=True, scaleFill=False, scaleUp=True,
                 vid_stride=1):
        """_summary_

        Args:
            sources (str, optional): _description_. Defaults to 'streams.txt'.
            img_size (int, optional): _description_. Defaults to 640.
            stride (int, optional): _description_. Defaults to 32.
            auto (bool, optional): _description_. Defaults to True.
        """

        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleUp = scaleUp
        self.vid_stride = vid_stride
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            if isinstance(sources, str):
                sources = [sources]
        n = len(sources)
        self.imgs, self.frames, self.threads = [None] * n, [0] * n, [None] * n
        self.c_frame = [0] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            if 'rtsp://' in s:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

            print(f'{i + 1}/{n}: {s}... init ')
            url = eval(s) if s.isnumeric() else s
            if urlparse(s).hostname in YOUTUBE:  # if source is YouTube video
                # check_requirements(('pafy', 'youtube_dl'))
                import pafy
                try:
                    url = pafy.new(url).getbest(preftype="mp4").url
                except Exception as ex:
                    # logger.error(f'if the error come from pafy library, please report to https://github.com/thnak/pafy.git')
                    # logger.info('attempting install pafy from git')
                    print(f"{ex}")
                    # os.system('pip install git+https://github.com/thnak/pafy.git')
            cap = cv2.VideoCapture(url, cv2.CAP_ANY)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, url]), daemon=True)
            print(f'success ({w}x{h} at {self.fps:.2f} FPS, {self.frames[i]} frames).')
            self.threads[i].start()
        print('')  # newline

    def update(self, index, cap, stream):
        # Read next stream frame in a daemon thread
        self.c_frame[index], f = 0, self.frames[index]
        while cap.isOpened() and self.c_frame[index] < f:
            self.c_frame[index] += 1
            cap.grab()
            if self.c_frame[index] % self.vid_stride == 0:  # read every 4th frame
                success, im = cap.retrieve()
                if success:
                    self.imgs[index] = im
                else:
                    self.imgs[index] = np.zeros_like(self.imgs[index])
                    cap.open(stream)
                    print('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
            if self.fps != 0:
                time.sleep(1 / self.fps)  # wait time
            else:
                time.sleep(1 / 30)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads):
            raise StopIteration
        img0 = self.imgs.copy()
        # Letterbox
        img, ratio, dwdh = [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[0]
                            for x in img0], \
            [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[1] for x in
             img0], \
            [letterbox(x, self.img_size, auto=self.auto, scaleFill=self.scaleFill, stride=self.stride)[2] for x in
             img0],
        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, {'frames': self.frames, 'c_frame': self.c_frame}, ratio, dwdh

    def __len__(self):
        return len(self.sources)