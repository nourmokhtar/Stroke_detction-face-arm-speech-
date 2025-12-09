import os
import torch
import pandas as pd
import sounddevice as sd
import soundfile as sf

# Adjust these imports according to dysarthria-mtl repo structure
from model import DysarthriaMTL
from inference import predict_from_csv  # or the repo's function to predict from CSV

# ----------------------------
# Step 1: Record user audio
# ----------------------------
fs = 16000  # sample rate
duration = 5  # seconds
filename = "patient.wav"

print(f"Please speak for {duration} seconds...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
sf.write(filename, audio, fs)
print(f"Audio saved as {filename}")

# ----------------------------
# Step 2: Create temporary CSV for the model
# ----------------------------
csv_path = "temp_test.csv"
df = pd.DataFrame({
    "name": ["patient_001"],
    "path": [os.path.abspath(filename)],
    "category": [0],  # dummy, repo expects it
    "text": ["Sample sentence"],  # optional
    "split": ["test"]
})
df.to_csv(csv_path, index=False)
print(f"CSV saved as {csv_path}")

# ----------------------------
# Step 3: Load pretrained model
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DysarthriaMTL()
weights_path = "pretrained_weights.pt"  # place your downloaded weights here
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

# ----------------------------
# Step 4: Run inference
# ----------------------------
results = predict_from_csv(csv_path, model, device=device)  # check repo signature
print("Predicted severity (0-4):", results)
