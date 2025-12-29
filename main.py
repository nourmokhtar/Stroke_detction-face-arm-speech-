# main.py
import numpy as np
import time

# Optional: for colored terminal output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = ""
        YELLOW = ""
        GREEN = ""
    class Style:
        RESET_ALL = ""

# -------------------------------
# Import your face & speech scripts
# -------------------------------
from face.test import run_face_inference   # should return dict with 'face_severity' & 'face_description'
from speech.test_voice import run_speech_inference  # should return dict with 'speech_label' & 'speech_probabilities'

# -------------------------------
# LLM-style reasoning
# -------------------------------
def llm_reasoning(face_result, speech_result):
    severity = face_result["face_severity"]
    face_desc = face_result["face_description"]
    speech_label = speech_result["speech_label"]
    speech_probs = speech_result["speech_probabilities"]

    # Deterministic fusion rules
    if severity >= 5 and speech_label == "Dysarthric":
        risk = "HIGH"
        color = Fore.RED
        explanation = (
            f"Face severity is {severity:.1f} ({face_desc}) AND speech is Dysarthric "
            f"(prob={speech_probs['Dysarthric']:.2f}) → HIGH risk."
        )
    elif severity >= 5 or speech_label == "Dysarthric":
        risk = "MODERATE"
        color = Fore.YELLOW
        explanation = (
            f"Face severity is {severity:.1f} ({face_desc}) OR speech is {speech_label} "
            f"(prob={speech_probs['Dysarthric']:.2f}) → MODERATE risk."
        )
    else:
        risk = "LOW"
        color = Fore.GREEN
        explanation = (
            f"Face severity is {severity:.1f} ({face_desc}) AND speech is {speech_label} "
            f"(prob={speech_probs['Dysarthric']:.2f}) → LOW risk."
        )

    return {"final_risk": risk, "explanation": explanation, "color": color}

# -------------------------------
# Logging helper
# -------------------------------
def log_section(title):
    print(f"\n{'='*10} {title} {'='*10}\n")

# -------------------------------
# Main
# -------------------------------
def main():
    log_section("FACE ANALYSIS")
    start_time = time.time()
    face_result = run_face_inference()
    elapsed = time.time() - start_time
    print(f"Face Severity Score : {face_result['face_severity']:.1f}")
    print(f"Face Description    : {face_result['face_description']}")
    print(f"[INFO] Face analysis completed in {elapsed:.2f}s")

    log_section("SPEECH ANALYSIS")
    start_time = time.time()
    speech_result = run_speech_inference()
    elapsed = time.time() - start_time
    print(f"Speech Result       : {speech_result['speech_label']}")
    print(f"Speech Probabilities: Control={speech_result['speech_probabilities']['Control']:.2f}, "
          f"Dysarthric={speech_result['speech_probabilities']['Dysarthric']:.2f}")

    print(f"[INFO] Speech analysis completed in {elapsed:.2f}s")

    log_section("FUSION & FINAL DECISION")
    final_result = llm_reasoning(face_result, speech_result)
    print(final_result["color"] + f"FINAL RISK LEVEL: {final_result['final_risk']}" + Style.RESET_ALL)
    print(f"Explanation       : {final_result['explanation']}")


if __name__ == "__main__":
    main()
