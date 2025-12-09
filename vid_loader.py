import cv2
import os
import numpy as np


def load_video(video_path, frame_skip=5):
    """
    Load a video and extract frames with a sampling strategy.
    
    Args:
        video_path (str): Path to the video file
        frame_skip (int): Number of frames to skip between samples
    
    Returns:
        frames (list of np.array): List of extracted frames
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only keep every nth frame
        if frame_count % frame_skip == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames


video_path = "taycir.mp4"  # Replace with your test video path
frames = load_video(video_path, frame_skip=5)
print(f"Number of frames extracted: {len(frames)}")

# Show the first frame to check
cv2.imshow("First Frame", frames[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


output_dir = "taycirr" 
os.makedirs(output_dir, exist_ok=True)

for i, frame in enumerate(frames):
    cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), frame)
