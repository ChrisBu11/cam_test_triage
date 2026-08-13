"""FairFace ethnicity inference from an ONNX export."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


FAIRFACE_LABELS = ("White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern")


class FairFaceEstimator:
    def __init__(self, model_path: str = "Emotion_Detection/models/fairface.onnx", history_size: int = 15):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing FairFace model: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.history: deque[str] = deque(maxlen=history_size)

    def predict(self, crop: np.ndarray) -> str:
        image = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: image.transpose(2, 0, 1)[None]})[0][0]
        label = FAIRFACE_LABELS[int(np.argmax(output))]
        self.history.append(label)
        return max(set(self.history), key=self.history.count)
