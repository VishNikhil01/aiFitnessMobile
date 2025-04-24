import os
import warnings
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import torch
import pickle
import random
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp


mp_dir = os.path.dirname(mp.__file__)
heavy_file = os.path.join(mp_dir, "modules/pose_landmark/pose_landmark_heavy.tflite")
if os.path.exists(heavy_file):
    try:
        os.chmod(heavy_file, 0o444)
    except PermissionError:
        pass

# ─── Suppress FutureWarnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Streamlit Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time AI Fitness Posture Correction (Mobile)",
    layout="centered",
    initial_sidebar_state="auto",
)

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.title("Settings")
enable_audio = st.sidebar.checkbox("Enable Audio Feedback", value=True)
exercise = st.sidebar.selectbox("Select Exercise", ["Bench Press", "Squat", "Deadlift"])
confidence_threshold = st.sidebar.slider("Landmark Tracking Confidence", 0.0, 1.0, 0.7)
counter_display = st.sidebar.empty()

# ─── Load YOLOv5 Person Detector ─────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    weights = "models/best_big_bounding.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    st.sidebar.write(f"YOLO running on: {device}")
    return model

yolo_model = load_yolo()

# ─── Load Exercise Posture Classifier ─────────────────────────────────────────
@st.cache_resource
def load_posture_model():
    paths = {
        "Bench Press": "models/benchpress/benchpress.pkl",
        "Squat":       "models/squat/squat.pkl",
        "Deadlift":    "models/deadlift/deadlift.pkl",
    }
    with open(paths[exercise], "rb") as f:
        return pickle.load(f)

pose_classifier = load_posture_model()

# ─── Utility Functions ───────────────────────────────────────────────────────
def most_frequent(lst):
    return max(lst, key=lst.count)

def calculateAngle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180.0 / np.pi)
    return 360.0 - ang if ang > 180.0 else ang

def detect_objects(frame):
    results = yolo_model(frame)
    return results.pred[0]



# ─── Feedback Options ─────────────────────────────────────────────────────────
FEEDBACK_OPTIONS = {
    "excessive_arch": [
        ("Avoid arching your lower back too much; keep it neutral.", "resources/sounds/excessive_arch_1.mp3"),
        ("Tighten your core to reduce lower back curvature.", "resources/sounds/excessive_arch_2.mp3"),
    ],
    "arms_spread": [
        ("Grip too wide; bring hands shoulder-width apart.", "resources/sounds/arms_spread_1.mp3"),
        ("Maintain a slightly narrower grip for stability.", "resources/sounds/arms_spread_2.mp3"),
    ],
    "spine_neutral": [
        ("Keep your spine neutral by lifting chest up.", "resources/sounds/spine_neutral_feedback_1.mp3"),
        ("Retract shoulders and straighten your back.", "resources/sounds/spine_neutral_feedback_2.mp3"),
    ],
    "caved_in_knees": [
        ("Do not let knees collapse inward; push them out.", "resources/sounds/caved_in_knees_feedback_1.mp3"),
        ("Focus on spreading knees in line with toes.", "resources/sounds/caved_in_knees_feedback_2.mp3"),
    ],
    "feet_spread": [
        ("Narrow your stance to shoulder width.", "resources/sounds/feet_spread.mp3"),
        ("Step feet closer for better balance.", "resources/sounds/feet_spread_2.mp3"),
    ],
    "arms_narrow": [
        ("Grip too narrow; widen slightly.", "resources/sounds/arms_narrow.mp3"),
    ],
}

# ─── Video Transformer for WebRTC ────────────────────────────────────────────
class FitnessTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.status = []
        self.last_alert = 0

        # MediaPipe Pose with complexity=1 (full model, not lite)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
            model_complexity=1  # <-- full model, avoids heavy-file permission issues
        )

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        detections = detect_objects(img)
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf = map(int, det[:5])
                if conf < 0.7:
                    continue

                roi = img[y1:y2, x1:x2]
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)
                if not res.pose_landmarks:
                    continue

                lm = res.pose_landmarks.landmark
                feats = [v for pt in lm for v in (pt.x, pt.y, pt.z, pt.visibility)]
                df = pd.DataFrame([feats])
                try:
                    pred = pose_classifier.predict(df)[0]
                except:
                    pred = "unknown"

                # Rep counting logic
                if "down" in pred:
                    self.stage = "down"
                    self.status.append(pred)
                elif self.stage == "down" and "up" in pred:
                    self.stage = "up"
                    self.counter += 1
                    counter_display.write(f"{exercise} Reps: {self.counter}")
                    tag = most_frequent(self.status)
                    if tag != "correct":
                        now = time.time()
                        if now - self.last_alert > 3:
                            msg, snd = random.choice(FEEDBACK_OPTIONS.get(tag, [("Adjust posture.", "")]))
                            st.error(msg)
                            if enable_audio and snd:
                                st.audio(snd)
                            self.last_alert = now
                    self.status.clear()
                elif "correct" in most_frequent(self.status):
                    st.success("Correct posture!")
                    if enable_audio:
                        st.audio("resources/sounds/correct.mp3")
                    self.status.clear()

                # Draw landmarks
                for lm_id in self.mp_pose.PoseLandmark:
                    if lm[lm_id.value].visibility >= confidence_threshold:
                        mp.solutions.drawing_utils.draw_landmarks(
                            roi,
                            res.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS
                        )
                img[y1:y2, x1:x2] = roi

        # Overlay rep count
        cv2.putText(img, f"Reps: {self.counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ─── Launch WebRTC Stream ───────────────────────────────────────────────────
st.title("🏋️ Real-Time AI Workout Form Correction (Mobile)")
st.markdown("Open this page in Safari (iOS) or Chrome (Android), allow camera access, and start your workout.")

webrtc_streamer(
    key="fitness_app",
    video_transformer_factory=FitnessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
