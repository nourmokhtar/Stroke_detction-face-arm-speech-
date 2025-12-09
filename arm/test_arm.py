import cv2
import mediapipe as mp

# Load image
frame = cv2.imread("extracted_frames_arm/frame_0044.jpg")
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

results = pose.process(rgb_frame)

# Arm landmark indices
# These are Mediapipe Pose landmark IDs
key_points = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16
}

if results.pose_landmarks:
    h, w, _ = frame.shape
    landmarks = results.pose_landmarks.landmark
    
    for name, idx in key_points.items():
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)  # green circle
        cv2.putText(frame, name, (x+5, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1)

cv2.imshow("Arm Landmarks", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
