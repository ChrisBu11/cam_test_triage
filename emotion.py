"""HSEmotion inference with confidence and temporal smoothing."""

from __future__ import annotations

from collections import deque
import os

import numpy as np

# HSEmotion distributes a trusted full-model checkpoint. PyTorch 2.6+
# changed torch.load's default to weights_only=True, which cannot deserialize
# this checkpoint. Set this before HSEmotion constructs the model.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionEstimator:
    def __init__(self, history_size: int = 5):
        self.model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        self.history: deque[tuple[str, float]] = deque(maxlen=history_size)

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        emotion, scores = self.model.predict_emotions(crop, logits=False)
        scores = np.asarray(scores, dtype=np.float32)
        confidence = float(np.max(scores) * 100.0) if scores.size else 0.0
        self.history.append((str(emotion), confidence))
        label = max(set(item[0] for item in self.history), key=lambda item: sum(x[0] == item for x in self.history))
        matching = [score for name, score in self.history if name == label]
        return label, float(np.mean(matching))
