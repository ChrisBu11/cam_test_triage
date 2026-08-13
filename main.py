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
from tracker import FacePresenceTracker

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

    # Initialize the identity database tracker
    face_tracker = FacePresenceTracker()
    
    # Recognition clock state configuration
    RECOGNITION_INTERVAL_SECS = 10.0
    last_recognition_time = 0.0
    current_user_id = None
    current_total_time = 0.0
    faces = []

    with Camera(config) as camera:
        for frame_number, frame in enumerate(camera):
            
            # --- FIX 1: Initialize these variables right at the start of EVERY frame loop ---
            heart_rate = None
            signal: list[float] = []
            
            # Check if it's time for a 10-second identification run
            now = time.time()
            should_run_recognition = (now - last_recognition_time >= RECOGNITION_INTERVAL_SECS)
            
            # Run tracking loop (Skip optimization frames but force a fresh scan on identification markers)
            if frame_number % 3 == 0 or should_run_recognition:
                faces = detector.detect(frame, run_recognition=should_run_recognition)
                
            primary = faces[0] if faces else None
            
            if primary is not None:
                # Passive Time Accumulation
                if current_user_id is not None:
                    current_total_time = face_tracker.update_passive_time(current_user_id)
                
                # Process the identity assignment using the pre-calculated flag
                if should_run_recognition:
                    last_recognition_time = now
                    try:
                        embedding = None
                        if hasattr(primary, 'normed_embedding') and primary.normed_embedding is not None:
                            embedding = primary.normed_embedding
                        elif hasattr(primary, 'embedding') and primary.embedding is not None:
                            embedding = primary.embedding
                            
                        if embedding is not None:
                            current_user_id, current_total_time = face_tracker.match_or_register(embedding)
                            print(f"[TRACKER] Matched ID: {current_user_id}")
                        else:
                            print("[TRACKER] Embedding blank - Keep head centered to camera lens.")
                    except Exception as rec_exc:
                        logging.warning("Identity check failed: %s", rec_exc)

                try:
                    landmarks = mesh.detect(frame, primary)
                    heart_rate = rppg.update(frame, landmarks, time.time())
                    signal = rppg.get_signal()
                except Exception as exc:
                    logging.warning("Face mesh/rPPG update failed: %s", exc)

                if frame_number % INFERENCE_INTERVAL == 0:
                    try:
                        emotion_name, confidence = emotion.predict(primary.crop)
                        
                        id_str = f"ID: {current_user_id}" if current_user_id else "ID: Scanning..."
                        time_str = f"{current_total_time:.1f}s"
                        
                        cached = {
                            "bbox": primary.bbox, 
                            "label": f"{id_str} ({time_str}) | {emotion_name} ({confidence:.0f}%) | {age.predict(primary)} yrs | {race.predict(primary.crop)}"
                        }
                    except Exception as exc:
                        logging.exception("Demographic inference failed: %s", exc)
            else:
                # Clear active tracking state if the user leaves the lens completely
                current_user_id = None
            
            # --- RENDERING PLOT LAYOUT (Safe from NameErrors now!) ---
            if cached is not None:
                draw_face(frame, cached["bbox"], str(cached["label"]), [], None)
                
                bbox = cached["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                box_width = x2 - x1
                
                # Draw bottom background vitals strip
                hr_text = f"Pulse: {heart_rate:.1f} BPM" if heart_rate is not None else "Pulse: Analyzing..."
                (t_width, t_height), _ = cv2.getTextSize(hr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y2 + 5), (x1 + t_width + 10, y2 + 30), (0, 0, 0), -1)
                
                text_color = (0, 255, 0) if heart_rate is not None else (0, 165, 255)
                cv2.putText(frame, hr_text, (x1 + 5, y2 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
                
                # Signal Graph rendering block follows below...
                if signal and len(signal) > 1:
                    graph_top = y2 + 35
                    graph_height = 40
                    graph_bottom = graph_top + graph_height
                    
                    cv2.rectangle(frame, (x1, graph_top), (x2, graph_bottom), (15, 15, 15), -1)
                    cv2.rectangle(frame, (x1, graph_top), (x2, graph_bottom), (50, 50, 50), 1)
                    
                    sig_min, sig_max = min(signal), max(signal)
                    sig_range = sig_max - sig_min if (sig_max - sig_min) > 1e-6 else 1.0
                    
                    points = []
                    num_samples = len(signal)
                    for i, val in enumerate(signal):
                        pt_x = x1 + int((i / (num_samples - 1)) * box_width)
                        norm_val = (val - sig_min) / sig_range
                        pt_y = graph_bottom - int(norm_val * (graph_height - 10)) - 5
                        points.append((pt_x, pt_y))
                    
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i+1], (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Facial Analysis + rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    run()
