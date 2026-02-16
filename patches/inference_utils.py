import av
import os
from fractions import Fraction
from queue import Queue
from threading import Thread

import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image


def _get_stream_frame_count(stream) -> int:
    """Best-effort frame count from stream metadata (avoids full decode when possible)."""
    if stream.duration is not None and stream.time_base is not None and stream.average_rate is not None:
        try:
            dur_sec = float(stream.duration * stream.time_base)
            rate = float(stream.average_rate)
            if dur_sec > 0 and rate > 0:
                return max(1, int(dur_sec * rate))
        except (TypeError, ZeroDivisionError):
            pass
    return 0


class VideoReader(Dataset):
    """Decode with PyAV FRAME/AUTO threading; store frames in one contiguous array for cache-friendly access."""

    def __init__(self, path, transform=None):
        self.transform = transform
        self._rate = 30.0
        container = av.open(path)
        if len(container.streams.video) == 0:
            container.close()
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        self._rate = float(stream.average_rate) if stream.average_rate else 30.0
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format="rgb24"))
        container.close()
        self._frames = np.stack(frames_list, axis=0) if frames_list else np.zeros((0, 1, 1, 3), dtype=np.uint8)

    @property
    def frame_rate(self):
        return self._rate

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        frame = self._frames[idx]
        frame = Image.fromarray(frame)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class StreamingVideoReader(IterableDataset):
    """Producer-consumer: decode in background thread while consumer runs RVM. Overlaps decode with inference."""

    def __init__(self, path, transform=None, queue_size=32):
        self.path = path
        self.transform = transform
        self._queue_size = queue_size
        self._rate = 30.0
        self._total_frames = 0
        container = av.open(path)
        if len(container.streams.video) == 0:
            container.close()
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        self._rate = float(stream.average_rate) if stream.average_rate else 30.0
        self._total_frames = _get_stream_frame_count(stream)
        self._container = container
        self._stream = stream
        if self._total_frames == 0:
            container.close()
            self._container = None

    @property
    def frame_rate(self):
        return self._rate

    def __len__(self):
        return self._total_frames if self._total_frames > 0 else 0

    def _decoder_thread(self, queue: Queue) -> None:
        if self._container is None:
            queue.put(None)
            return
        try:
            for frame in self._container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                queue.put(arr)
            queue.put(None)
        finally:
            self._container.close()

    def __iter__(self):
        if self._total_frames == 0:
            return
        queue = Queue(maxsize=self._queue_size)
        t = Thread(target=self._decoder_thread, args=(queue,))
        t.start()
        while True:
            arr = queue.get()
            if arr is None:
                break
            frame = Image.fromarray(arr)
            if self.transform is not None:
                frame = self.transform(frame)
            yield frame
        t.join()


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        rate = Fraction.from_float(float(frame_rate))
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=rate)
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate

    def write(self, frames):
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1)
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class ImageSequenceReader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = sorted(os.listdir(path))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)

    def write(self, frames):
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1

    def close(self):
        pass
