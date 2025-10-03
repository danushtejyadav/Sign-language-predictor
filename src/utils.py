import mediapipe as mp
import numpy as np

# --- MediaPipe Initialization ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks(results):
    """
    Extracts hand and face landmarks from MediaPipe results into a single array.
    Pads with zeros if landmarks are not detected.
    """
    landmarks = []
    
    # Left hand landmarks (21 landmarks * 3 coordinates)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Right hand landmarks (21 landmarks * 3 coordinates)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # A subset of face landmarks for efficiency (20 landmarks * 3 coordinates)
    face_indices = [0, 4, 8, 12, 14, 17, 21, 33, 37, 40, 43, 46, 49, 55, 69, 105, 127, 132, 148, 152]
    if results.face_landmarks:
        for idx in face_indices:
            lm = results.face_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (len(face_indices) * 3))
    
    return np.array(landmarks)