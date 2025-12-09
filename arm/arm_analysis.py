# arm_mobility_front.py

import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# ===============================
# Paths
# ===============================
frames_dir = "extracted_frames_arm"  # folder with your frames
output_csv = "arm_features.csv"      # output CSV

# ===============================
# Mediapipe setup
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# ===============================
# Helper functions
# ===============================
def get_angle(a, b, c):
    """Calculate the angle (degrees) between three points a-b-c"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def to_px(lm, w, h):
    return np.array([int(lm.x * w), int(lm.y * h)])

# ===============================
# Prepare CSV
# ===============================
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_wrist_height",
        "right_wrist_height",
        "wrist_height_diff",
        "weak_arm_flag"
    ])

# ===============================
# Process frames
# ===============================
for frame_file in sorted(os.listdir(frames_dir)):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Left arm
        left_sh = to_px(lm[11], w, h)
        left_el = to_px(lm[13], w, h)
        left_wr = to_px(lm[15], w, h)
        left_angle = get_angle(left_sh, left_el, left_wr)
        left_wrist_height = left_wr[1]

        # Right arm
        right_sh = to_px(lm[12], w, h)
        right_el = to_px(lm[14], w, h)
        right_wr = to_px(lm[16], w, h)
        right_angle = get_angle(right_sh, right_el, right_wr)
        right_wrist_height = right_wr[1]

        # Wrist height difference (absolute vertical difference)
        wrist_diff = abs(left_wrist_height - right_wrist_height)

        # Weak arm flag: either elbow angle < 70% or wrist drift > threshold
        weak_arm = int(left_angle < 126 or right_angle < 126 or wrist_diff > 20)

        # Save to CSV
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_file,
                left_angle,
                right_angle,
                left_wrist_height,
                right_wrist_height,
                wrist_diff,
                weak_arm
            ])

print(f"Front-arm raise features saved to {output_csv}")
