import os
import cv2
import time
import json
import argparse
import numpy as np
from utils import holistic, extract_landmarks

# --- Configuration ---
DATA_PATH = "data/raw"
SEQUENCE_LENGTH = 3  # Seconds
FPS = 15
FRAMES_PER_SEQUENCE = SEQUENCE_LENGTH * FPS

def collect_data(sign_name, num_samples):
    """
    Captures and saves sign language data samples.
    """
    sign_path = os.path.join(DATA_PATH, sign_name)
    os.makedirs(sign_path, exist_ok=True)
    
    # Find the starting sample number
    start_sample = 0
    existing_files = [f for f in os.listdir(sign_path) if f.endswith('.json')]
    if existing_files:
        start_sample = max([int(f.split('.')[0]) for f in existing_files]) + 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    for i in range(start_sample, start_sample + num_samples):
        print(f"\n--- Collecting sample {i} for sign '{sign_name}' ---")
        
        # Countdown
        for j in range(3, 0, -1):
            print(f"Get ready... {j}")
            time.sleep(1)
        
        print("Recording...")
        
        sequence_data = []
        start_time = time.time()
        
        while len(sequence_data) < FRAMES_PER_SEQUENCE:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Extract landmarks
            landmarks = extract_landmarks(results)
            sequence_data.append(landmarks.tolist())
            
            # Display progress on frame
            progress_text = f"Sample {i} | Frame {len(sequence_data)}/{FRAMES_PER_SEQUENCE}"
            cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)

            # Control FPS
            time.sleep(max(0, (1.0/FPS) - (time.time() - start_time)))
            start_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Collection stopped by user.")
                return

        # Save the collected sequence
        sample_file = os.path.join(sign_path, f"{i}.json")
        with open(sample_file, 'w') as f:
            json.dump(sequence_data, f)
            
        print(f"Successfully saved sample {i} to {sample_file}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n--- Finished collecting {num_samples} samples for '{sign_name}' ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sign language data.")
    parser.add_argument("--sign", type=str, required=True, help="Name of the sign to collect data for.")
    parser.add_argument("--samples", type=int, required=True, help="Number of samples to collect.")
    
    args = parser.parse_args()
    
    collect_data(args.sign, args.samples)