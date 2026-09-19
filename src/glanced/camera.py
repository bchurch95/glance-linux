"""V4L2 capture.

Two frames come out of every grab, and the distinction matters to liveness:

* the **full frame**, which the bezel detector needs because it has to look
  *around* the face for a device edge, not just at it;
* a **native-resolution crop** around the face for the gloss/glare cue, which
  only ever downsamples — never upsamples — so `GlareSample.crop_pixel_width`
  stays an honest measure of how much real detail was available.
"""

from __future__ import annotations

import fcntl
import glob
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

log = logging.getLogger("glanced.camera")

#: The resolution the liveness tuning constants assume. Several gates
#: (`min_yaw_range_degrees`, `MIN_RELIABLE_INTEROCULAR_PX`) are expressed in
#: pixels at this width; changing it silently invalidates them.
WORKING_WIDTH = 640

VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001


def find_paired_depth_camera(rgb_device: str = "/dev/video0") -> Optional[str]:
    """Find a paired 3D depth or active IR camera associated with `rgb_device`.

    On machines with hardware depth/IR sensors (such as Acer laptops equipped
    with the Quanta / Acer FHD + IR camera module, Intel RealSense, or Windows
    Hello IR sensors), the color camera and depth/IR camera share the same USB
    parent device.
    """
    try:
        rgb_real = os.path.realpath(rgb_device)
        rgb_name = os.path.basename(rgb_real)
        sysfs_path = f"/sys/class/video4linux/{rgb_name}"
        if not os.path.exists(sysfs_path):
            return None

        device_dir = os.path.realpath(os.path.join(sysfs_path, "device"))
        usb_parent = os.path.dirname(device_dir)

        candidates = []
        for v in sorted(glob.glob("/sys/class/video4linux/video*")):
            v_name = os.path.basename(v)
            if v_name == rgb_name:
                continue
            v_dev = os.path.realpath(os.path.join(v, "device"))
            v_parent = os.path.dirname(v_dev)

            # Must share the same USB hub or controller parent
            if v_parent != usb_parent and os.path.dirname(v_parent) != os.path.dirname(usb_parent):
                continue

            name_file = os.path.join(v, "name")
            card_name = open(name_file).read().strip() if os.path.exists(name_file) else ""

            # Check for IR / Depth / Spatial / 3D camera keywords
            keywords = ("ir", "depth", "spatial", "3d")
            if not any(kw in card_name.lower() for kw in keywords):
                continue

            dev_path = f"/dev/{v_name}"
            # Verify video capture capability (filter out metadata capture nodes)
            try:
                fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
                cap = bytearray(104)
                fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
                _, _, _, _, _, device_caps = struct.unpack("16s32s32sIII", cap[:92])
                os.close(fd)
                if device_caps & V4L2_CAP_VIDEO_CAPTURE:
                    candidates.append((dev_path, card_name))
            except OSError:
                continue

        if candidates:
            log.info("found paired depth/IR camera: %s (%s)", candidates[0][0], candidates[0][1])
            return candidates[0][0]
    except Exception as e:
        log.debug("error searching for paired depth camera: %s", e)
    return None


@dataclass
class FramePair:
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: Optional[str] = "MJPG"
    #: Depth/IR camera node, "auto" to discover paired depth camera, or None to disable.
    depth_device: Optional[str] = "auto"
    depth_width: int = 640
    depth_height: int = 360
    depth_fps: int = 15


class _DepthReader:
    """Asynchronous reader for secondary depth/IR camera.

    Prevents slower depth sensors (typically 15fps) from blocking or stuttering
    the primary 30fps RGB pipeline.
    """

    def __init__(self, device: str, width: int = 640, height: int = 360, fps: int = 15) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: Optional["cv2.VideoCapture"] = None
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            log.warning("could not open depth camera %s", self.device)
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="DepthReader")
        self._thread.start()
        return True

    def _run(self) -> None:
        while self._running and self._capture is not None:
            ok, bgr = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class Camera:
    def __init__(self, config: CameraConfig = CameraConfig()) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for capture")
        self.config = config
        self._capture: Optional["cv2.VideoCapture"] = None
        self._depth_reader: Optional[_DepthReader] = None
        self.resolved_depth_device: Optional[str] = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        capture = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"could not open {self.config.device}")
        if self.config.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture = capture

        # Resolve depth device
        depth_target = self.config.depth_device
        if depth_target == "auto":
            depth_target = find_paired_depth_camera(self.config.device)

        if depth_target:
            reader = _DepthReader(
                depth_target,
                width=self.config.depth_width,
                height=self.config.depth_height,
                fps=self.config.depth_fps,
            )
            if reader.start():
                self._depth_reader = reader
                self.resolved_depth_device = depth_target
                log.info("active depth camera: %s", depth_target)

    def close(self) -> None:
        if self._depth_reader is not None:
            self._depth_reader.stop()
            self._depth_reader = None
        self.resolved_depth_device = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield RGB frames until the device stops delivering."""
        for pair in self.frame_pairs():
            yield pair.rgb

    def frame_pairs(self) -> Iterator[FramePair]:
        """Yield FramePairs with RGB and concurrent depth/IR frames."""
        if self._capture is None:
            raise RuntimeError("camera is not open")
        while True:
            ok, bgr = self._capture.read()
            if not ok:
                return
            now = time.monotonic()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = self._depth_reader.latest_frame() if self._depth_reader else None
            yield FramePair(rgb=rgb, depth=depth, timestamp=now)


def to_working_resolution(frame: np.ndarray, width: int = WORKING_WIDTH) -> tuple[np.ndarray, float]:
    """Downscale to the working width. Returns the frame and the scale factor
    that maps working-resolution coordinates back to native pixels."""
    height, native_width = frame.shape[:2]
    if native_width <= width:
        return frame, 1.0
    scale = width / native_width
    resized = cv2.resize(frame, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def render_crop(
    frame: np.ndarray, bounding_box: Sequence[float], padding: float = 0.15
) -> Optional[np.ndarray]:
    """Native-resolution crop around a face box, with a little padding.

    Never upsamples: if the requested region is smaller than the box asks for,
    what comes back is what the sensor actually resolved. The gloss/glare cue
    confidence-weights on the returned width precisely so that an
    under-resolved crop abstains instead of guessing.
    """
    height, width = frame.shape[:2]
    x, y, w, h = (float(v) for v in bounding_box)
    pad_x, pad_y = w * padding, h * padding
    x0 = int(max(0, round(x - pad_x)))
    y0 = int(max(0, round(y - pad_y)))
    x1 = int(min(width, round(x + w + pad_x)))
    y1 = int(min(height, round(y + h + pad_y)))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]
