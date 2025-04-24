import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import time
import pygame
import torch
import pickle
import random

# ─── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-time Big Three Workout AI Posture Correction Service",
    layout="centered",
)

# ─── Load models ───────────────────────────────────────────────────────────────
# YOLOv5
YOLO_WEIGHTS = r"D:/ISL-MAJOR-PROJECT/models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=YOLO_WEIGHTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Cache exercise posture classifier
@st.cache_resource
def load_posture_model(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# ─── Utility functions ────────────────────────────────────────────────────────

def most_frequent(data):
    return max(data, key=data.count)


def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def detect_objects(frame):
    results = model(frame)
    return results.pred[0]

# ─── App UI ───────────────────────────────────────────────────────────────────
st.title("Real-time Big Three Workout AI Posture Correction Service")

menu = st.sidebar.selectbox(
    "Select Exercise",
    ["Bench Press", "Squat", "Deadlift"]
)
counter_display = st.sidebar.empty()
confidence_slider = st.sidebar.slider(
    "Joint Detection Confidence Threshold", 0.0, 1.0, 0.7
)

# Map menu choice to posture model path
POSTURE_PATHS = {
    "Bench Press": r"D:/ISL-MAJOR-PROJECT/models/benchpress/benchpress.pkl",
    "Squat":       r"D:/ISL-MAJOR-PROJECT/models/Sqat/squat.pkl",
    "Deadlift":    r"D:/ISL-MAJOR-PROJECT/models/deadlift/deadlift.pkl",
}
model_e = load_posture_model(POSTURE_PATHS[menu])

# Initialize counters and state
pygame.mixer.init()
counter = 0
current_stage = ""
posture_status = [None]
counter_display.header(f"Current Counter: {counter} reps")

# Video frame display
FRAME_WINDOW = st.image([])

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=2
)

# ─── Main camera loop ─────────────────────────────────────────────────────────
def run_camera():
    global counter, current_stage, posture_status
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        preds = detect_objects(frame)
        try:
            if preds is not None:
                for det in preds:
                    c1, c2 = det[:2].int(), det[2:4].int()
                    conf = float(det[4])
                    if conf < 0.7:
                        continue

                    x1, y1 = c1[0].item(), c1[1].item()
                    x2, y2 = c2[0].item(), c2[1].item()
                    roi = frame[y1:y2, x1:x2]

                    # Pose estimation
                    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    res = pose.process(img_rgb)
                    if not res.pose_landmarks:
                        continue
                    lm = res.pose_landmarks.landmark

                    # Extract landmark coordinates
                    pts = {
                        'nose':    [lm[mp_pose.PoseLandmark.NOSE].x,    lm[mp_pose.PoseLandmark.NOSE].y],
                        'l_sh':    [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                        'l_el':    [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,    lm[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                        'l_wr':    [lm[mp_pose.PoseLandmark.LEFT_WRIST].x,    lm[mp_pose.PoseLandmark.LEFT_WRIST].y],
                        'l_hip':   [lm[mp_pose.PoseLandmark.LEFT_HIP].x,     lm[mp_pose.PoseLandmark.LEFT_HIP].y],
                        'l_kn':    [lm[mp_pose.PoseLandmark.LEFT_KNEE].x,    lm[mp_pose.PoseLandmark.LEFT_KNEE].y],
                        'l_an':    [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x,   lm[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                        'l_he':    [lm[mp_pose.PoseLandmark.LEFT_HEEL].x,    lm[mp_pose.PoseLandmark.LEFT_HEEL].y],
                        'r_sh':    [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                        'r_el':    [lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,    lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                        'r_wr':    [lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,    lm[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                        'r_hip':   [lm[mp_pose.PoseLandmark.RIGHT_HIP].x,     lm[mp_pose.PoseLandmark.RIGHT_HIP].y],
                        'r_kn':    [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x,    lm[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                        'r_an':    [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x,   lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
                        'r_he':    [lm[mp_pose.PoseLandmark.RIGHT_HEEL].x,    lm[mp_pose.PoseLandmark.RIGHT_HEEL].y],
                    }

                    # Angle calculations
                    neck_angle = (
                        calculateAngle(pts['l_sh'], pts['nose'], pts['l_hip']) +
                        calculateAngle(pts['r_sh'], pts['nose'], pts['r_hip'])
                    ) / 2
                    left_elbow_angle    = calculateAngle(pts['l_sh'], pts['l_el'], pts['l_wr'])
                    right_elbow_angle   = calculateAngle(pts['r_sh'], pts['r_el'], pts['r_wr'])
                    left_shoulder_angle = calculateAngle(pts['l_el'], pts['l_sh'], pts['l_hip'])
                    right_shoulder_angle= calculateAngle(pts['r_el'], pts['r_sh'], pts['r_hip'])
                    left_hip_angle      = calculateAngle(pts['l_sh'], pts['l_hip'], pts['l_kn'])
                    right_hip_angle     = calculateAngle(pts['r_sh'], pts['r_hip'], pts['r_kn'])
                    left_knee_angle     = calculateAngle(pts['l_hip'], pts['l_kn'], pts['l_an'])
                    right_knee_angle    = calculateAngle(pts['r_hip'], pts['r_kn'], pts['r_an'])
                    left_ankle_angle    = calculateAngle(pts['l_kn'], pts['l_an'], pts['l_he'])
                    right_ankle_angle   = calculateAngle(pts['r_kn'], pts['r_an'], pts['r_he'])

                    # Update sidebar angles
                    st.sidebar.text(f"Neck Angle: {neck_angle:.2f}°")
                    st.sidebar.text(f"L Elbow: {left_elbow_angle:.2f}°")
                    st.sidebar.text(f"R Elbow: {right_elbow_angle:.2f}°")
                    st.sidebar.text(f"L Shoulder: {left_shoulder_angle:.2f}°")
                    st.sidebar.text(f"R Shoulder: {right_shoulder_angle:.2f}°")
                    st.sidebar.text(f"L Hip: {left_hip_angle:.2f}°")
                    st.sidebar.text(f"R Hip: {right_hip_angle:.2f}°")
                    st.sidebar.text(f"L Knee: {left_knee_angle:.2f}°")
                    st.sidebar.text(f"R Knee: {right_knee_angle:.2f}°")
                    st.sidebar.text(f"L Ankle: {left_ankle_angle:.2f}°")
                    st.sidebar.text(f"R Ankle: {right_ankle_angle:.2f}°")

                    # Rep counting
                    try:
                        feats = [coord for lm_ in lm for coord in (lm_.x, lm_.y, lm_.z, lm_.visibility)]
                        df = pd.DataFrame([feats])
                        pred = model_e.predict(df)[0]

                        if "down" in pred:
                            current_stage = "down"
                            posture_status.append(pred)
                        elif current_stage == "down" and "up" in pred:
                            current_stage = "up"
                            counter += 1
                            counter_display.header(f"Current Counter: {counter} reps")

                            if "correct" not in most_frequent(posture_status):
                                now = time.time()
                                if now - previous_alert_time >= 3:
                                    common = most_frequent(posture_status)
                                    # Feedback messages
                                    feedbacks = {
                                        'excessive_arch': [
                                            ("Avoid arching your lower back too much; keep it neutral.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/excessive_arch_1.mp3"),
                                            ("Lift your pelvis slightly and engage your core to flatten your back.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/excessive_arch_2.mp3"),
                                        ],
                                        'arms_spread': [
                                            ("Your grip is too wide. Narrow it slightly.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/arms_spread_1.mp3"),
                                            ("Hold the bar just wider than shoulder width.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/arms_spread_2.mp3"),
                                        ],
                                        'spine_neutral': [
                                            ("Avoid excessive spine curvature.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/spine_neutral_feedback_1.mp3"),
                                            ("Lift your chest and retract your shoulders.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/spine_neutral_feedback_2.mp3"),
                                        ],
                                        'caved_in_knees': [
                                            ("Don't let your knees cave in during squats.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/caved_in_knees_feedback_1.mp3"),
                                            ("Push your hips back to align knees and toes.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/caved_in_knees_feedback_2.mp3"),
                                        ],
                                        'feet_spread': [
                                            ("Narrow your stance to shoulder width.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/feet_spread.mp3"),
                                        ],
                                        'arms_narrow': [
                                            ("Your grip is too wide. Narrow it slightly.", r"D:/ISL-MAJOR-PROJECT/resources/sounds/arms_narrow.mp3"),
                                        ],
                                    }
                                    if common in feedbacks:
                                        msg, snd = random.choice(feedbacks[common])
                                        st.error(msg)
                                        pygame.mixer.music.load(snd)
                                        pygame.mixer.music.play()
                                        posture_status = []
                                        previous_alert_time = now
                    except Exception:
                        pass

                    # Draw landmarks & bbox
                    mp.solutions.drawing_utils.draw_landmarks(
                        roi, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame[y1:y2, x1:x2] = roi

        except Exception:
            pass

        FRAME_WINDOW.image(frame)
    cap.release()

# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_camera()
