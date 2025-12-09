import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# ===============================
# Paths
# ===============================
frames_dir = "taycirr"          # Your input frames
output_csv = "face_features_v9.csv"       # Upgraded output

# ===============================
# Mediapipe setup
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,        # Important: gives iris landmarks + better precision
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# Prepare CSV with meaningful, stroke-relevant columns
# ===============================
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "smile_vertical_asymmetry_px",   # KEY stroke indicator
        "mouth_horizontal_asymmetry_px",
        "eye_horizontal_asymmetry_px",
        "general_symmetry_score",
        "facial_droop_flag",             # 1 = probable droop
        "severity_score"                 # 0-100 (for final decision engine)
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
        (33, 263), (263, 33),   # outer eye corners
        (61, 291), (291, 61),   # mouth corners
        (362, 133), (133, 362), # inner eye corners (with refine_landmarks)
        (234, 454), (454, 234), # cheeks
        (93, 323),  (323, 93),  # jaw
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

    # Key landmarks (standard Face Mesh indices)
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
    # 1. Smile Vertical Asymmetry → THE #1 stroke sign when smiling!
    # ===============================
    smile_vertical_asymmetry = abs(left_mouth_px[1] - right_mouth_px[1])
    # Lower y = higher on screen → one side not lifting = stroke

    # ===============================
    # 2. Horizontal symmetry (distance from midline)
    # ===============================
    mouth_h_asym = abs( abs(left_mouth_px[0] - nose_px[0]) - abs(right_mouth_px[0] - nose_px[0]) )
    eye_h_asym   = abs( abs(left_eye_px[0]   - nose_px[0]) - abs(right_eye_px[0]   - nose_px[0]) )

    # ===============================
    # 3. General face symmetry score
    # ===============================
    general_symmetry_score = compute_general_symmetry(lm, w, h)

    # ===============================
    # 4. Facial droop detection flag (tuned on real stroke vs normal)
    # ===============================
    droop_detected = (
        smile_vertical_asymmetry > 28 or    # Very strong indicator
        mouth_h_asym > 22 or
        general_symmetry_score > 18
    )
    facial_droop_flag = 1 if droop_detected else 0

    # ===============================
    # 5. Severity score (0–100) – perfect for final decision engine
    # ===============================
    severity = 0
    severity += min(smile_vertical_asymmetry / 50 * 50, 50)   # max 50 points
    severity += min(mouth_h_asym / 40 * 30, 30)               # max 30 points
    severity += min(general_symmetry_score / 30 * 20, 20)     # max 20 points
    severity = min(severity, 100)

    # ===============================
    # Save to CSV
    # ===============================
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_file,
            round(smile_vertical_asymmetry, 2),
            round(mouth_h_asym, 2),
            round(eye_h_asym, 2),
            round(general_symmetry_score, 2),
            facial_droop_flag,
            round(severity, 2)
        ])

    # Optional: visualize on frame (uncomment to debug)
cv2.putText(frame, f"Smile Asym: {smile_vertical_asymmetry:.1f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Face Analysis", frame)
cv2.waitKey(1)

print(f"Enhanced face analysis complete! → {output_csv}")
print("   Key improvement: smile_vertical_asymmetry = real stroke smile test!")