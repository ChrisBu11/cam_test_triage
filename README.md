# Facial Analysis + rPPG Webcam Application

The modular implementation requested in `TASK.md` is [`main.py`](main.py).
It uses InsightFace for multi-face detection, HSEmotion for emotion, DEX for
age, FairFace for ethnicity, and the existing POS implementation in
[`rppg.py`](rppg.py). See [`MODEL_DOWNLOADS.md`](MODEL_DOWNLOADS.md) before
running it.

This project combines:

- OpenCV webcam capture
- DeepFace emotion, age, gender, and race analysis
- Real-time rPPG heart-rate estimation from a forehead ROI

## Files

- [`Face_Detection.py`](C:/Users/z3352157/OneDrive%20-%20UNSW/Desktop/git_repos/facedetect/Emotion_Detection/Face_Detection.py)
- [`rppg.py`](C:/Users/z3352157/OneDrive%20-%20UNSW/Desktop/git_repos/facedetect/Emotion_Detection/rppg.py)

## How It Works

1. The webcam frame is captured with OpenCV.
2. DeepFace runs on the frame every `INFERENCE_INTERVAL` frames.
3. MediaPipe Tasks detects facial landmarks for the forehead ROI.
4. The forehead RGB signal is buffered in `RPPGProcessor`.
5. POS + Butterworth bandpass filtering + FFT estimate heart rate.
6. The latest BPM is drawn under the DeepFace labels.

## Dependencies

Install these into the project virtual environment:

```bash
pip install opencv-python deepface mediapipe scipy numpy
```

If DeepFace pulls additional backend packages in your environment, keep them installed as well.

## MediaPipe Model

Place the landmark model next to the scripts:

```text
Emotion_Detection/face_landmarker.task
```

The code expects:

```python
FACE_LANDMARK_MODEL_PATH = "face_landmarker.task"
```

## Run

```bash
python Face_Detection.py
```

Press `q` to exit.

## Notes

- DeepFace inference is intentionally throttled with `INFERENCE_INTERVAL` so the webcam stays responsive.
- rPPG estimation runs continuously and only updates the displayed HR about once per second.
- The script keeps running if no face is detected, the forehead ROI is unavailable, or HR cannot be estimated yet.
