from collections import deque
import os
import time

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARK_MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
FOREHEAD_LANDMARK_INDICES = (10, 54, 67, 103, 151, 284, 297, 332, 338)


def _as_pixel_point(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


class FaceLandmarkDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path is not None else FACE_LANDMARK_MODEL_PATH
        self._detector = None

    def initialize(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Missing MediaPipe model file: {self.model_path}"
            )

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # FIX: Switch running mode to VIDEO for real-time temporal tracking (Massive FPS boost)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, 
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        return self

    def detect(self, frame, *args):
        if self._detector is None:
            self.initialize()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        
        timestamp_ms = int(time.time() * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result or not result.face_landmarks:
            return []

        # MediaPipe returns a list of faces. We must return the landmarks directly 
        # so that rppg.py's indexing loop can read individual points.
        return result.face_landmarks[0]


def get_forehead_roi(frame, landmarks):
    if not landmarks:
        return None

    frame_height, frame_width = frame.shape[:2]

    points = []
    for index in FOREHEAD_LANDMARK_INDICES:
        if index < len(landmarks):
            points.append(_as_pixel_point(landmarks[index], frame_width, frame_height))

    if len(points) < 3:
        return None

    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.float32)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)

    # Expand upward toward the hairline and keep the lower edge above the eyes.
    x1 = int(max(0, min_x - 0.18 * width))
    x2 = int(min(frame_width, max_x + 0.18 * width))
    y1 = int(max(0, min_y - 0.65 * height))
    y2 = int(min(frame_height, min_y + 0.25 * height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _estimate_heart_rate_with_signal(rgb_buffer, fps):
    samples = np.asarray(rgb_buffer, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] != 3:
        return None, None

    min_samples = max(int(fps * 5), 60)
    if samples.shape[0] < min_samples:
        return None, None

    channel_means = samples.mean(axis=0)
    if np.any(channel_means <= 1e-6):
        return None, None

    normalized = samples / channel_means - 1.0

    # POS projection: convert RGB into a pulse-sensitive chrominance signal.
    projection = np.array(
        [[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    projected = projection @ normalized.T
    std_0 = float(np.std(projected[0]))
    std_1 = float(np.std(projected[1]))
    if std_1 <= 1e-6:
        return None, None

    alpha = std_0 / std_1 if std_0 > 1e-6 else 0.0
    pulse = projected[0] + alpha * projected[1]
    pulse = pulse - np.mean(pulse)

    if np.std(pulse) <= 1e-6:
        return None, None

    nyquist = 0.5 * fps
    low = 0.5 / nyquist
    high = 3.0 / nyquist
    if high >= 1.0:
        return None, None

    b, a = butter(3, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, pulse)
    if np.std(filtered) <= 1e-6:
        return None, None

    window = np.hanning(filtered.size)
    spectrum = np.abs(np.fft.rfft(filtered * window))
    frequencies = np.fft.rfftfreq(filtered.size, d=1.0 / fps)
    valid = (frequencies >= 0.5) & (frequencies <= 3.0)
    if not np.any(valid):
        return None, filtered

    valid_spectrum = spectrum[valid]
    if valid_spectrum.size == 0:
        return None, filtered

    peak_index = int(np.argmax(valid_spectrum))
    peak_power = float(valid_spectrum[peak_index])
    if peak_power <= 1e-6 or peak_power < float(np.mean(valid_spectrum)) * 1.15:
        return None, filtered

    peak_frequency = float(frequencies[valid][peak_index])
    heart_rate_bpm = peak_frequency * 60.0

    if heart_rate_bpm < 30.0 or heart_rate_bpm > 180.0:
        return None, filtered

    return heart_rate_bpm, filtered


def estimate_heart_rate(rgb_buffer, fps):
    heart_rate_bpm, _ = _estimate_heart_rate_with_signal(rgb_buffer, fps)
    return heart_rate_bpm


class RPPGProcessor:
    def __init__(self, fps=30, buffer_seconds=20, smoothing_window=5, estimate_interval=1.0):
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer = deque(maxlen=int(fps * buffer_seconds))
        self.signal_buffer = deque(maxlen=int(fps * 15))
        self.hr_history = deque(maxlen=smoothing_window)
        self.estimate_interval = estimate_interval
        self.last_estimate_time = 0.0
        self.latest_hr_bpm = None

    def add_sample(self, rgb_sample):
        sample = np.asarray(rgb_sample, dtype=np.float32)
        if sample.shape != (3,):
            return False
        self.buffer.append(sample)
        return True

    def add_forehead_sample(self, frame, landmarks):
        roi = get_forehead_roi(frame, landmarks)
        if roi is None:
            return False

        x1, y1, x2, y2 = roi
        forehead = frame[y1:y2, x1:x2]
        if forehead.size == 0:
            return False

        rgb_forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
        mean_rgb = rgb_forehead.reshape(-1, 3).mean(axis=0)
        return self.add_sample(mean_rgb)

    def estimate(self, now=None):
        if now is None:
            now = time.time()

        if now - self.last_estimate_time < self.estimate_interval:
            return self.latest_hr_bpm

        self.last_estimate_time = now
        heart_rate, filtered_signal = _estimate_heart_rate_with_signal(self.buffer, self.fps)
        if filtered_signal is not None:
            self.signal_buffer.clear()
            self.signal_buffer.extend(float(value) for value in filtered_signal[-self.signal_buffer.maxlen :])

        if heart_rate is None:
            return self.latest_hr_bpm

        self.hr_history.append(float(heart_rate))
        self.latest_hr_bpm = float(np.mean(self.hr_history))
        return self.latest_hr_bpm

    def update(self, frame, landmarks, now=None):
        self.add_forehead_sample(frame, landmarks)
        return self.estimate(now=now)

    def get_signal(self):
        return list(self.signal_buffer)
