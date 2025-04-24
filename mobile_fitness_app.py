import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import random
import torch
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configure Streamlit page for mobile-friendly layout
st.set_page_config(
    page_title="Real-Time AI Fitness Posture Correction (Mobile)",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load YOLOv5 model for person detection
MODEL_WEIGHTS = "./models/best_big_bounding.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_WEIGHTS)
# Pick the best available device
if torch.backends.mps.is_available():       
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)
st.sidebar.write(f"Running on: {device}")
 # Use 'mps' on supported devices
model.eval()

# Utility: most frequent element

def most_frequent(data):
    return max(data, key=data.count)

# Compute angle between three points a-b-c

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Perform YOLOv5 object detection on a frame

def detect_objects(frame):
    results = model(frame)
    return results.pred[0]

# Sidebar controls
enable_audio = st.sidebar.checkbox("Enable Audio Feedback", value=True)
menu_selection = st.sidebar.selectbox(
    "Select Exercise", ("Bench Press", "Squat", "Deadlift")
)
counter_display = st.sidebar.empty()
confidence_threshold = st.sidebar.slider(
    "Landmark Tracking Confidence", min_value=0.0, max_value=1.0, value=0.7
)

# Mapping of feedback messages and audio files for common posture issues
FEEDBACK_OPTIONS = {
    "excessive_arch": [
        ("Avoid arching your lower back too much; keep it neutral.", "./resources/sounds/excessive_arch_1.mp3"),
        ("Tighten your core to reduce lower back curvature.", "./resources/sounds/excessive_arch_2.mp3"),
    ],
    "arms_spread": [
        ("Your grip is too wide; bring hands shoulder-width apart.", "./resources/sounds/arms_spread_1.mp3"),
        ("Maintain a slightly narrower grip for stability.", "./resources/sounds/arms_spread_2.mp3"),
    ],
    "spine_neutral": [
        ("Keep your spine neutral by lifting chest up.", "./resources/sounds/spine_neutral_feedback_1.mp3"),
        ("Retract shoulders and straighten your back.", "./resources/sounds/spine_neutral_feedback_2.mp3"),
    ],
    "caved_in_knees": [
        ("Do not let knees collapse inward; push them out.", "./resources/sounds/caved_in_knees_feedback_1.mp3"),
        ("Focus on spreading knees in line with toes.", "./resources/sounds/caved_in_knees_feedback_2.mp3"),
    ],
    "feet_spread": [
        ("Narrow your stance to about shoulder width.", "./resources/sounds/feet_spread.mp3"),
        ("Step feet closer for better balance.", "./resources/sounds/feet_spread_2.mp3"),
    ],
    "arms_narrow": [
        ("Grip is too narrow; widen slightly.", "./resources/sounds/arms_narrow.mp3"),
    ],
}

# Define video transformer for WebRTC streaming
class FitnessTransformer(VideoTransformerBase):
    """
    Video frame processor for real-time posture correction.
    """
    def __init__(self):
        # Rep counter and posture tracking
        self.counter = 0
        self.current_stage = ""
        self.posture_status = []
        self.previous_alert_time = 0

        # Load exercise-specific classifier
        model_path = {
            "Bench Press": "./models/benchpress/benchpress.pkl",
            "Squat": "./models/squat/squat.pkl",
            "Deadlift": "./models/deadlift/deadlift.pkl",
        }[menu_selection]
        with open(model_path, "rb") as f:
            self.model_e = pickle.load(f)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
            model_complexity=2,
        )

    def recv(self, frame):
        """
        Processes each video frame: detection, pose estimation,
        angle computation, classification, counting, feedback.
        """
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror for user

        # Detect persons
        detections = detect_objects(img)
        for det in detections:
            c1, c2 = det[:2].int(), det[2:4].int()
            conf = float(det[4])
            if conf < 0.7:
                continue

            x1, y1 = c1[0].item(), c1[1].item()
            x2, y2 = c2[0].item(), c2[1].item()
            person_frame = img[y1:y2, x1:x2]

            # Pose estimation
            rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(rgb)
            if not results_pose.pose_landmarks:
                continue
            lm = results_pose.pose_landmarks.landmark

            # Extract keypoints
            pts = {p.name: [lm[p].x, lm[p].y] for p in self.mp_pose.PoseLandmark}

            # Calculate angles
            neck_angle = (
                calculateAngle(pts['LEFT_SHOULDER'], pts['NOSE'], pts['LEFT_HIP']) +
                calculateAngle(pts['RIGHT_SHOULDER'], pts['NOSE'], pts['RIGHT_HIP'])
            ) / 2
            angles = {
                'left_elbow': calculateAngle(pts['LEFT_SHOULDER'], pts['LEFT_ELBOW'], pts['LEFT_WRIST']),
                'right_elbow': calculateAngle(pts['RIGHT_SHOULDER'], pts['RIGHT_ELBOW'], pts['RIGHT_WRIST']),
                'left_shoulder': calculateAngle(pts['LEFT_ELBOW'], pts['LEFT_SHOULDER'], pts['LEFT_HIP']),
                'right_shoulder': calculateAngle(pts['RIGHT_ELBOW'], pts['RIGHT_SHOULDER'], pts['RIGHT_HIP']),
                'left_hip': calculateAngle(pts['LEFT_SHOULDER'], pts['LEFT_HIP'], pts['LEFT_KNEE']),
                'right_hip': calculateAngle(pts['RIGHT_SHOULDER'], pts['RIGHT_HIP'], pts['RIGHT_KNEE']),
                'left_knee': calculateAngle(pts['LEFT_HIP'], pts['LEFT_KNEE'], pts['LEFT_ANKLE']),
                'right_knee': calculateAngle(pts['RIGHT_HIP'], pts['RIGHT_KNEE'], pts['RIGHT_ANKLE']),
                'left_ankle': calculateAngle(pts['LEFT_KNEE'], pts['LEFT_ANKLE'], pts['LEFT_HEEL']),
                'right_ankle': calculateAngle(pts['RIGHT_KNEE'], pts['RIGHT_ANKLE'], pts['RIGHT_HEEL']),
            }

            # Sidebar angle display on mobile
            st.sidebar.write(f"Neck Angle: {neck_angle:.2f}°")
            for name, value in angles.items():
                st.sidebar.write(f"{name.replace('_',' ').title()} Angle: {value:.2f}°")

            # Classification and counting
            try:
                row = [coord for lm_pt in lm for coord in [lm_pt.x, lm_pt.y, lm_pt.z, lm_pt.visibility]]
                X = pd.DataFrame([row])
                exercise_class = self.model_e.predict(X)[0]

                if "down" in exercise_class:
                    self.current_stage = "down"
                    self.posture_status.append(exercise_class)
                elif self.current_stage == "down" and "up" in exercise_class:
                    self.current_stage = "up"
                    self.counter += 1
                    counter_display.write(f"Count: {self.counter}")
                    self.posture_status.append(exercise_class)

                    # Provide feedback for incorrect posture
                    tag = most_frequent(self.posture_status)
                    if tag != "correct":
                        now = time.time()
                        if now - self.previous_alert_time >= 3:
                            message, audio = random.choice(
                                FEEDBACK_OPTIONS.get(tag, [("Adjust posture.", "")])
                            )
                            st.error(message)
                            if enable_audio and audio:
                                st.audio(audio)
                            self.posture_status.clear()
                            self.previous_alert_time = now
                elif "correct" in most_frequent(self.posture_status):
                    st.success("Correct posture!")
                    if enable_audio:
                        st.audio("./resources/sounds/correct.mp3")
                    self.posture_status.clear()
            except Exception:
                pass

            # Draw landmarks (optional debug overlay)
            for lm_id in self.mp_pose.PoseLandmark:
                if lm[lm_id.value].visibility >= confidence_threshold:
                    mp.solutions.drawing_utils.draw_landmarks(
                        person_frame,
                        results_pose.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
            img[y1:y2, x1:x2] = person_frame

        # Overlay rep count on final image
        cv2.putText(
            img,
            f"Reps: {self.counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Start WebRTC streamer for mobile browsers
webrtc_streamer(
    key="fitness-webrtc",
    video_transformer_factory=FitnessTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Note: Ensure camera permission is enabled in your Android/iPhone browser settings.
