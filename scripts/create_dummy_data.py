import os
import json

# --- Configuration ---
NUM_FEATURES = 186
NUM_FRAMES = 45
DATA_PATH = "data/raw"
SIGNS = ["SIGN_A", "SIGN_B"]
SAMPLES_PER_SIGN = 2

def generate_dummy_data():
    """Creates dummy data files to allow the CI pipeline to run train.py."""
    
    print("Generating dummy data for CI test...")
    
    # Create a single frame of zeros
    frame = [0.0] * NUM_FEATURES
    
    # Create a sequence of frames (the full data for one sample)
    sequence = [frame] * NUM_FRAMES
    
    for sign in SIGNS:
        sign_path = os.path.join(DATA_PATH, sign)
        os.makedirs(sign_path, exist_ok=True)
        
        for i in range(SAMPLES_PER_SIGN):
            sample_file = os.path.join(sign_path, f"{i}.json")
            with open(sample_file, 'w') as f:
                json.dump(sequence, f)

    print(f"Successfully created {SAMPLES_PER_SIGN} dummy samples for signs: {', '.join(SIGNS)}")

if __name__ == "__main__":
    generate_dummy_data()