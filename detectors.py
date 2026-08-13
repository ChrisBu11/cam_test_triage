"""InsightFace detection and primary-face MediaPipe landmark handling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from rppg import FaceLandmarkDetector, FACE_LANDMARK_MODEL_PATH


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    crop: np.ndarray


class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l', 
            allowed_modules=['detection', 'genderage', 'recognition'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    # FIX 1: Add run_recognition=False flag to conditionally toggle heavy embedding calculations
    def detect(self, frame: np.ndarray, run_recognition: bool = False) -> list[DetectedFace]:
        if not hasattr(self, 'app'):
            self.__init__()
            
        # Dynamically toggle the recognition model on or off to optimize performance
        if not run_recognition:
            self.app.models['recognition'].max_num = 0  # Skips embedding code path
        else:
            self.app.models['recognition'].max_num = 1  # Computes embedding vector
            
        height, width = frame.shape[:2]
        result: list[DetectedFace] = []
        
        for face in self.app.get(frame):
            x1, y1, x2, y2 = [int(value) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
                
            # FIX 2: Safely extract embeddings from InsightFace if they exist on this frame
            normed_emb = getattr(face, 'normed_embedding', None)
            raw_emb = getattr(face, 'embedding', None)
            
            # Create your custom object
            detected_face_obj = DetectedFace((x1, y1, x2, y2), float(face.det_score), frame[y1:y2, x1:x2].copy())
            
            # FIX 3: Inject the embedding attribute directly onto your custom dataclass object
            detected_face_obj.normed_embedding = normed_emb
            detected_face_obj.embedding = raw_emb
            
            result.append(detected_face_obj)
            
        return sorted(result, key=lambda item: item.confidence, reverse=True)


class PrimaryFaceMesh:
    # 2. Change the default argument value to None
    def __init__(self, model_path: str = None):
        # 3. Fall back to the absolute path variable if no path was passed
        if model_path is None:
            model_path = FACE_LANDMARK_MODEL_PATH
            
        self.detector = FaceLandmarkDetector(model_path)
        self.detector.initialize()

    def detect(self, frame: np.ndarray, face: DetectedFace):
        x1, y1, x2, y2 = face.bbox
        landmarks = self.detector.detect(face.crop)
        # rppg.py consumes normalized landmarks. Translate crop coordinates back
        # to full-frame coordinates while retaining normalized landmark objects.
        crop_h, crop_w = face.crop.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        return [
            SimpleNamespace(
                x=(x1 + landmark.x * crop_w) / frame_w,
                y=(y1 + landmark.y * crop_h) / frame_h,
                z=landmark.z,
            )
            for landmark in landmarks
        ]
