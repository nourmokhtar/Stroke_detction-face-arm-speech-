import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# ===============================
# Paths
# ===============================
frames_dir = r"C:\Users\Lenovo-Thinkpad\Desktop\4AI\projet s1\strokedetection\dataimages\data\face\Image\1"     
output_csv ="nostroke_ieee.csv"       


# ===============================
# Mediapipe setup
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,        # gives iris landmarks + better precision
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# Prepare CSV with stroke-relevant columns
# ===============================
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "smile_vertical_asymmetry_norm",
        "mouth_horizontal_asymmetry_norm",
        "eye_horizontal_asymmetry_norm",
        "general_symmetry_score_norm",
        "facial_droop_flag",
        "severity_score"
    ])

# ===============================
# Helper functions
# ===============================
def to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def compute_general_symmetry(landmarks, img_w, img_h):
    nose = to_px(landmarks[1], img_w, img_h)  # nose tip
    diffs = []
    pairs = [
        (33, 263), (61, 291), (362, 133), (234, 454), (93, 323)
    ]
    for l_idx, r_idx in pairs:
        left = to_px(landmarks[l_idx], img_w, img_h)
        right = to_px(landmarks[r_idx], img_w, img_h)
        diffs.append(abs(abs(left[0] - nose[0]) - abs(right[0] - nose[0])))
    return np.mean(diffs) if diffs else 0

# ===============================
# Process each frame
# ===============================
for frame_file in sorted(os.listdir(frames_dir)):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        continue

    lm = results.multi_face_landmarks[0].landmark

    # Key landmarks
    nose_tip       = lm[1]
    left_mouth     = lm[61]
    right_mouth    = lm[291]
    left_eye_outer = lm[33]
    right_eye_outer= lm[263]

    # Convert to pixels
    nose_px       = to_px(nose_tip, w, h)
    left_mouth_px = to_px(left_mouth, w, h)
    right_mouth_px= to_px(right_mouth, w, h)
    left_eye_px   = to_px(left_eye_outer, w, h)
    right_eye_px  = to_px(right_eye_outer, w, h)

    # ===============================
    # Face-size normalization factor
    # ===============================
    inter_ocular_distance = np.linalg.norm(left_eye_px - right_eye_px)

    # ===============================
    # 1. Smile Vertical Asymmetry
    # ===============================
    smile_vertical_asymmetry = abs(left_mouth_px[1] - right_mouth_px[1])
    smile_vertical_asymmetry_norm = smile_vertical_asymmetry / inter_ocular_distance

    # ===============================
    # 2. Horizontal asymmetry
    # ===============================
    mouth_h_asym = abs(abs(left_mouth_px[0] - nose_px[0]) - abs(right_mouth_px[0] - nose_px[0]))
    mouth_h_asym_norm = mouth_h_asym / inter_ocular_distance

    eye_h_asym = abs(abs(left_eye_px[0] - nose_px[0]) - abs(right_eye_px[0] - nose_px[0]))
    eye_h_asym_norm = eye_h_asym / inter_ocular_distance

    # ===============================
    # 3. General symmetry score
    # ===============================
    general_symmetry_score = compute_general_symmetry(lm, w, h)
    general_symmetry_score_norm = general_symmetry_score / inter_ocular_distance

    # ===============================
    # 4. Facial droop detection flag
    # ===============================
    droop_detected = (
        smile_vertical_asymmetry_norm > 0.28 or
        mouth_h_asym_norm > 0.22 or
        general_symmetry_score_norm > 0.18
    )
    facial_droop_flag = 1 if droop_detected else 0

    # ===============================
    # 5. Severity score (0–100)
    # ===============================
    severity = 0
    severity += min(smile_vertical_asymmetry_norm / 0.5 * 50, 50)
    severity += min(mouth_h_asym_norm / 0.4 * 30, 30)
    severity += min(general_symmetry_score_norm / 0.3 * 20, 20)
    severity = min(severity, 100)

    # ===============================
    # Save to CSV
    # ===============================
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_file,
            round(smile_vertical_asymmetry_norm, 4),
            round(mouth_h_asym_norm, 4),
            round(eye_h_asym_norm, 4),
            round(general_symmetry_score_norm, 4),
            facial_droop_flag,
            round(severity, 2)
        ])

    # Optional: visualize
    cv2.putText(frame, f"Severity: {severity:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Face Analysis", frame)
    cv2.waitKey(1)

print(f"Normalized face analysis complete! → {output_csv}")
