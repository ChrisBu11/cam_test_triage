"""Webcam capture helpers with multi-threaded lag mitigation."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    index: int = 1
    # OPTIMIZATION: Dropping resolution to 640x480 drastically reduces CPU load
    width: int = 640  
    height: int = 480  
    fps: int = 20


class Camera:
    """Multi-threaded, context-managed camera wrapper to eliminate frame lag."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def __enter__(self) -> "Camera":
        # DirectShow backend helps Windows load the webcam and controls faster
        self.capture = cv2.VideoCapture(self.config.index, cv2.CAP_DSHOW)
        
        # Force strict hardware constraints
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        if not self.capture.isOpened():
            # Fallback to default backend if DirectShow fails
            self.capture = cv2.VideoCapture(self.config.index)
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open webcam index {self.config.index}")

        # Start background consumer thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Give the webcam a moment to warm up and grab the first frame
        time.sleep(0.5)
        return self

    def _capture_loop(self) -> None:
        """Continuously pulls frames from the hardware buffer on a separate thread."""
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.capture is None:
            raise RuntimeError("Camera must be used as a context manager")
        while self.running:
            with self.lock:
                current_frame = self.frame.copy() if self.frame is not None else None
            
            if current_frame is not None:
                yield current_frame
            else:
                # Small sleep to prevent thread thrashing if frame isn't ready
                time.sleep(0.005)

    def __exit__(self, *_: object) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
