# ─── mobile_fitness_app.py ───────────────────────────────────────────────────

import os
# ─── Force SDL to use dummy audio/video drivers (for headless / Streamlit Cloud) ───
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import streamlit as st
import numpy as np
import pandas as pd
import time
import pygame
import torch
import pickle
import random

# ─── Disable Mediapipe downloader (avoid PermissionErrors) ───────────────────────
import mediapipe as mp
try:
    import mediapipe.python.solutions.download_utils as _dl
    _dl.download_oss_model = lambda *a, **k: None
    import mediapipe.python.solutions.pose as _pose_mod
    _pose_mod._download_oss_pose_landmark_model = lambda c: None
except Exception:
    pass

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ─── Use ultralytics instead of torch.hub ─────────────────────────────────────
#   pip install ultralytics
from ultralytics import YOLO

@st.cache_resource
def load_yolo():
    return YOLO("models/best_big_bounding.pt")

yolo = load_yolo()

# ─── Load posture classifier ─────────────────────────────────────────────────
@st.cache_resource
def load_posture_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ─── UI Controls 
st.set_page_config(
    page_title="Real-time Big Three Workout AI Posture Correction Service",
    layout="centered",
)
st.title("Real-time Big Three Workout AI Posture Correction Service")

exercise = st.sidebar.selectbox("Select Exercise", ["Bench Press", "Squat", "Deadlift"])
confidence_threshold = st.sidebar.slider("Landmark Confidence", 0.0, 1.0, 0.7)
counter_display = st.sidebar.empty()

POSTURE_PATHS = {
    "Bench Press": "models/benchpress/benchpress.pkl",
    "Squat":       "models/squat/squat.pkl",
    "Deadlift":    "models/deadlift/deadlift.pkl"
}
pose_model = load_posture_model(POSTURE_PATHS[exercise])

# ─── Pygame init for audio ───────────────────────────────────────────────────
mixer_available = True
try:
    pygame.mixer.init()
except pygame.error as e:
    mixer_available = False
    st.warning(f"Audio disabled: {e}")

# ─── Utility functions ──────────────────────────────────────────────────────
def most_frequent(lst):
    return max(lst, key=lst.count)

def calculateAngle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

def detect_objects(frame):
    # ultralytics: returns a list of results; each has .boxes.xyxy and .boxes.conf
    results = yolo.predict(source=frame, conf=confidence_threshold / 1.0, verbose=False)
    # take first (and only) result
    r = results[0]
    # xyxy: numpy array shape (n,4), conf: (n,)
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    return list(zip(boxes.tolist(), confs.tolist()))

# ─── Feedback options ───────────────────────────────────────────────────────
FEEDBACK = {
    "excessive_arch": [("Avoid arching your lower back too much; keep it neutral.", "sounds/excessive_arch_1.mp3")],
    "arms_spread":    [("Your grip is too wide; bring hands shoulder-width apart.","sounds/arms_spread_1.mp3")],
    "spine_neutral":  [("Keep your spine neutral by lifting chest.","sounds/spine_neutral_feedback_1.mp3")],
    "caved_in_knees":[("Do not let knees collapse inward.","sounds/caved_in_knees_feedback_1.mp3")],
    "feet_spread":    [("Narrow your stance to shoulder width.","sounds/feet_spread.mp3")],
    "arms_narrow":    [("Grip too narrow; widen slightly.","sounds/arms_narrow.mp3")],
}

# ─── WebRTC video transformer ────────────────────────────────────────────────
class Transformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = ""
        self.history = []
        self.last_alert = 0.0

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
            model_complexity=0    # light, no downloads
        )

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = cv2.flip(img_bgr, 1)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for (box, conf) in detect_objects(img_bgr):
            x1, y1, x2, y2 = map(int, box)
            if conf < confidence_threshold:
                continue

            roi = img[y1:y2, x1:x2]
            res = self.pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks.landmark

            pts = {p.name: [lm[p].x, lm[p].y] for p in mp.solutions.pose.PoseLandmark}

            # compute angles (neck, elbows, etc.)...
            neck = (
                calculateAngle(pts['LEFT_SHOULDER'], pts['NOSE'], pts['LEFT_HIP'])
              + calculateAngle(pts['RIGHT_SHOULDER'], pts['NOSE'], pts['RIGHT_HIP'])
            ) / 2
            le = calculateAngle(pts['LEFT_SHOULDER'], pts['LEFT_ELBOW'], pts['LEFT_WRIST'])
            re = calculateAngle(pts['RIGHT_SHOULDER'], pts['RIGHT_ELBOW'], pts['RIGHT_WRIST'])
            ls = calculateAngle(pts['LEFT_ELBOW'], pts['LEFT_SHOULDER'], pts['LEFT_HIP'])
            rs = calculateAngle(pts['RIGHT_ELBOW'], pts['RIGHT_SHOULDER'], pts['RIGHT_HIP'])
            lh = calculateAngle(pts['LEFT_SHOULDER'], pts['LEFT_HIP'], pts['LEFT_KNEE'])
            rh = calculateAngle(pts['RIGHT_SHOULDER'], pts['RIGHT_HIP'], pts['RIGHT_KNEE'])
            lk = calculateAngle(pts['LEFT_HIP'], pts['LEFT_KNEE'], pts['LEFT_ANKLE'])
            rk = calculateAngle(pts['RIGHT_HIP'], pts['RIGHT_KNEE'], pts['RIGHT_ANKLE'])
            la = calculateAngle(pts['LEFT_KNEE'], pts['LEFT_ANKLE'], pts['LEFT_HEEL'])
            ra = calculateAngle(pts['RIGHT_KNEE'], pts['RIGHT_ANKLE'], pts['RIGHT_HEEL'])

            # Sidebar stats
            sb = st.sidebar
            sb.text(f"Neck: {neck:.1f}°"); sb.text(f"L Elbow: {le:.1f}°")
            sb.text(f"R Elbow: {re:.1f}°"); sb.text(f"L Shoulder: {ls:.1f}°")
            sb.text(f"R Shoulder: {rs:.1f}°"); sb.text(f"L Hip: {lh:.1f}°")
            sb.text(f"R Hip: {rh:.1f}°"); sb.text(f"L Knee: {lk:.1f}°")
            sb.text(f"R Knee: {rk:.1f}°"); sb.text(f"L Ankle: {la:.1f}°")
            sb.text(f"R Ankle: {ra:.1f}°")

            # Rep counting
            feats = [v for pt in lm for v in (pt.x, pt.y, pt.z, pt.visibility)]
            df = pd.DataFrame([feats])
            pred = pose_model.predict(df)[0]

            if "down" in pred:
                self.stage = "down"; self.history.append(pred)
            elif self.stage=="down" and "up" in pred:
                self.stage="up"; self.counter+=1
                counter_display.header(f"{exercise} Reps: {self.counter}")
                tag = most_frequent(self.history)
                now = time.time()
                if tag!="correct" and now-self.last_alert>3:
                    msg,snd = random.choice(FEEDBACK.get(tag,[("Adjust posture.","")]))
                    st.error(msg)
                    if snd and mixer_available:
                        pygame.mixer.music.load(snd); pygame.mixer.music.play()
                    self.history.clear(); self.last_alert=now
            elif "correct" in most_frequent(self.history):
                st.success("Correct posture!")
                now=time.time()
                if now-self.last_alert>3 and mixer_available:
                    pygame.mixer.music.load("sounds/correct.mp3"); pygame.mixer.music.play()
                self.history.clear()

            # Draw box & landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                roi, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            img[y1:y2,x1:x2]=roi

        cv2.putText(img, f"Reps: {self.counter}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ─── Run WebRTC with back camera ─────────────────────────────────────────────
st.markdown("**Allow camera access. Uses back-facing camera on mobile.**")
webrtc_streamer(
    key="app",
    video_processor_factory=Transformer,
    media_stream_constraints={
        "video": {"facingMode": {"exact": "environment"}},
        "audio": False
    },
    async_processing=True
)
