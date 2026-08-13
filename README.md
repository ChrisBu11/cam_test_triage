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

you need to download and add fairface.onnx to the models folder


