"""CPU-first facial analysis and rPPG application."""

from __future__ import annotations

import logging
import time

import cv2

from age import DEXAgeEstimator
from camera import Camera, CameraConfig
from detectors import FaceDetector, PrimaryFaceMesh
from emotion import EmotionEstimator
from overlay import draw_face
from race import FairFaceEstimator
from rppg import RPPGProcessor, FaceLandmarkDetector, FACE_LANDMARK_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
INFERENCE_INTERVAL = 60


def run() -> None:
    detector = FaceDetector()
    mesh = FaceLandmarkDetector()
    emotion = EmotionEstimator()
    age = DEXAgeEstimator()
    race = FairFaceEstimator()
    config = CameraConfig()
    rppg = RPPGProcessor(fps=config.fps, buffer_seconds=20)
    cached: dict[str, object] | None = None

    with Camera(config) as camera:
        for frame_number, frame in enumerate(camera):
            
            # OPTIMIZATION: Only run heavy face detection once every 3 frames
            if frame_number % 3 == 0:
                faces = detector.detect(frame)
                
            primary = faces[0] if faces else None
            heart_rate = None
            signal: list[float] = []
            
            if primary is not None:
                try:
                    landmarks = mesh.detect(frame, primary)
                    heart_rate = rppg.update(frame, landmarks, time.time())
                    signal = rppg.get_signal()
                except Exception as exc:
                    logging.warning("Face mesh/rPPG update failed: %s", exc)

                if frame_number % INFERENCE_INTERVAL == 0:
                    try:
                        emotion_name, confidence = emotion.predict(primary.crop)
                        cached = {
                            "bbox": primary.bbox, 
                            "label": f"{emotion_name} ({confidence:.0f}%) | {age.predict(primary)} yrs | {race.predict(primary.crop)}"
                        }
                    except Exception as exc:
                        logging.exception("Demographic inference failed: %s", exc)
            
            if cached is not None:
                # 1. Draw original box and demographics label (at the top)
                draw_face(frame, cached["bbox"], str(cached["label"]), [], None)
                
                # 2. Extract bounding box coordinates [xmin, ymin, xmax, ymax]
                bbox = cached["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                box_width = x2 - x1
                
                # --- DRAW HEART RATE UNDER THE BOX ---
                hr_text = f"Pulse: {heart_rate:.1f} BPM" if heart_rate is not None else "Pulse: Analyzing..."
                text_pos = (x1, y2 + 25)
                
                # Draw a dark background banner for the text
                (t_width, t_height), _ = cv2.getTextSize(hr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + t_width + 10, y2 + 30), (0, 0, 0), -1)
                
                # Print the text (Green if active, Orange if calculating)
                text_color = (0, 255, 0) if heart_rate is not None else (0, 165, 255)
                cv2.putText(frame, hr_text, (x1 + 5, y2 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
                
                # --- DRAW THE SIGNAL GRAPH UNDER THE TEXT ---
                if signal and len(signal) > 1:
                    graph_top = y2 + 35
                    graph_height = 40
                    graph_bottom = graph_top + graph_height
                    
                    # Draw a dark background box for the graph spanning the width of the face box
                    cv2.rectangle(frame, (x1, graph_top), (x2, graph_bottom), (15, 15, 15), -1)
                    cv2.rectangle(frame, (x1, graph_top), (x2, graph_bottom), (50, 50, 50), 1) # Border
                    
                    # Normalize the signal to fit inside our mini graph height
                    sig_min, sig_max = min(signal), max(signal)
                    sig_range = sig_max - sig_min if (sig_max - sig_min) > 1e-6 else 1.0
                    
                    # Map signal array points to pixel coordinates
                    points = []
                    num_samples = len(signal)
                    for i, val in enumerate(signal):
                        # Spread points across the width of the bounding box
                        pt_x = x1 + int((i / (num_samples - 1)) * box_width)
                        # Scale value to fit the graph height (inverted for image coordinates)
                        norm_val = (val - sig_min) / sig_range
                        pt_y = graph_bottom - int(norm_val * (graph_height - 10)) - 5
                        points.append((pt_x, pt_y))
                    
                    # Draw the pulse line step-by-step
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i+1], (0, 255, 255), 2, cv2.LINE_AA) # Cyan wave

            cv2.imshow("Facial Analysis + rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    run()
