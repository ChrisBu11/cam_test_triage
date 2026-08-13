import time
from collections import deque

import cv2
import numpy as np
from deepface import DeepFace

try:
    from rppg import FaceLandmarkDetector, RPPGProcessor
except ImportError:
    FaceLandmarkDetector = None
    RPPGProcessor = None


CAPTURE_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 1
INFERENCE_INTERVAL = 15
INFERENCE_SCALE = 0.5
WINDOW_NAME = "Emotion Detection"
DISPLAY_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
age_history = deque(maxlen=50)


def normalize_analysis_result(analysis):
    if isinstance(analysis, dict):
        return [analysis]
    return analysis or []


def analyze_frame(frame):
    scale = INFERENCE_SCALE
    if scale != 1.0:
        inference_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        inference_frame = frame

    analysis = DeepFace.analyze(
        inference_frame,
        actions=["emotion", "age", "gender", "race"],
        enforce_detection=False,
        detector_backend="opencv",
    )
    return normalize_analysis_result(analysis), scale


def extract_face_annotations(analysis, frame_shape, scale):
    frame_h, frame_w = frame_shape[:2]
    annotations = []

    for face in analysis:
        region = face.get("region") or {}
        x = int(region.get("x", 0) / scale)
        y = int(region.get("y", 0) / scale)
        w = int(region.get("w", 0) / scale)
        h = int(region.get("h", 0) / scale)

        if w <= 0 or h <= 0:
            continue

        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        emotion = face.get("dominant_emotion", "unknown")
        age = face.get("age", "?")
        gender = face.get("dominant_gender", "?")
        race = face.get("dominant_race", "?")
        confidence = face.get("emotion", {}).get(emotion)

        smoothed_age = age
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            age_value = None

        if age_value is not None:
            age_history.append(age_value)
            smoothed_age = int(round(float(np.mean(age_history))))

        if confidence is not None:
            label = f"{emotion} ({confidence:.1f}%) | {gender}, {smoothed_age} yrs | {race}"
        else:
            label = f"{emotion} | {gender}, {smoothed_age} yrs | {race}"

        annotations.append(
            {
                "box": (x, y, w, h),
                "label": label,
            }
        )

    return annotations


def draw_rppg_waveform(frame, signal, x, y):
    if signal is None:
        return

    values = np.asarray(signal, dtype=np.float32)
    if values.size < 2:
        return

    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-6:
        return

    normalized = (values - mean) / std
    width = 400
    height = 120
    title_height = 18
    padding = 8

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x1 + width)
    y2 = min(frame.shape[0], y1 + height)
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        "rPPG waveform (15s)",
        (x1 + padding, y1 + title_height),
        FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    plot_top = y1 + title_height + padding
    plot_bottom = y2 - padding
    plot_left = x1 + padding
    plot_right = x2 - padding
    plot_height = max(1, plot_bottom - plot_top)
    plot_width = max(1, plot_right - plot_left)

    if normalized.size > plot_width:
        indices = np.linspace(0, normalized.size - 1, plot_width).astype(np.int32)
        values = normalized[indices]
    else:
        values = normalized

    if values.size < 2:
        return

    min_val = float(values.min())
    max_val = float(values.max())
    if abs(max_val - min_val) <= 1e-6:
        return

    points = []
    denom = max(1, values.size - 1)
    for i, value in enumerate(values):
        px = plot_left + int(i * plot_width / denom)
        normalized_y = (float(value) - min_val) / (max_val - min_val)
        py = plot_bottom - int(normalized_y * plot_height)
        points.append((px, py))

    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, (0, 255, 255), 2, cv2.LINE_AA)


def draw_multiline_label(frame, x, y, lines):
    rendered_lines = [line for line in lines if line]
    if not rendered_lines:
        return

    font_scale = 0.6
    line_gap = 6
    padding = 8
    sizes = [cv2.getTextSize(line, FONT, font_scale, THICKNESS) for line in rendered_lines]
    max_width = max(size[0][0] for size in sizes)
    total_height = sum(size[0][1] for size in sizes) + (len(sizes) - 1) * line_gap
    baseline = max(size[1] for size in sizes)

    left = max(0, x)
    top = max(0, y - total_height - baseline - padding * 2)
    right = min(frame.shape[1] - 1, left + max_width + padding * 2)
    bottom = min(frame.shape[0] - 1, top + total_height + baseline + padding * 2)

    cv2.rectangle(frame, (left, top), (right, bottom), TEXT_BG_COLOR, cv2.FILLED)

    cursor_y = top + padding + sizes[0][0][1]
    for line, size in zip(rendered_lines, sizes):
        cv2.putText(
            frame,
            line,
            (left + padding, cursor_y),
            FONT,
            font_scale,
            TEXT_COLOR,
            THICKNESS,
            cv2.LINE_AA,
        )
        cursor_y += size[0][1] + line_gap


def extract_faces_and_detect_emotions():
    cap = cv2.VideoCapture(CAPTURE_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    landmark_detector = None
    rppg = None
    if FaceLandmarkDetector is not None and RPPGProcessor is not None:
        try:
            landmark_detector = FaceLandmarkDetector()
            landmark_detector.initialize()
            rppg = RPPGProcessor(fps=30, buffer_seconds=20)
        except Exception as exc:
            print(f"rPPG disabled: {exc}")

    frame_count = 0
    cached_annotations = []
    cached_error = None
    latest_hr_bpm = None
    rppg_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        if landmark_detector is not None and rppg is not None:
            try:
                landmarks = landmark_detector.detect(frame)
                latest_hr_bpm = rppg.update(frame, landmarks or [], time.time())
                rppg_signal = rppg.get_signal()
            except Exception as exc:
                print(f"rPPG warning: {exc}")

        if frame_count % INFERENCE_INTERVAL == 0:
            try:
                analysis, scale = analyze_frame(frame)
                cached_annotations = extract_face_annotations(analysis, frame.shape, scale)
                cached_error = None
            except Exception as exc:
                cached_annotations = []
                cached_error = f"DeepFace error: {exc}"

        hr_text = f"HR: {latest_hr_bpm:.0f} bpm" if latest_hr_bpm is not None else "HR: -- bpm"

        for index, annotation in enumerate(cached_annotations):
            x, y, w, h = annotation["box"]
            label = annotation["label"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), DISPLAY_COLOR, THICKNESS)
            draw_multiline_label(
                frame,
                x,
                y,
                [label, hr_text] if index == 0 else [label],
            )
            if index == 0:
                draw_rppg_waveform(frame, rppg_signal, x, y + h + 20)

        if cached_error:
            cv2.putText(
                frame,
                cached_error,
                (20, 40),
                FONT,
                0.7,
                (0, 0, 255),
                THICKNESS,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    extract_faces_and_detect_emotions()
