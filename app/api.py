import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- Configuration ---
MODELS_PATH = "models"
MODEL_NAME = "sign_model.h5"
ENCODER_NAME = "label_encoder.pkl"
FRAMES_PER_SEQUENCE = 45 # This must match the training configuration (3s * 15fps)

# --- FastAPI App Initialization ---
app = FastAPI(title="Sign Language Recognition API")

# --- Model and Encoder Loading ---
# Load artifacts at startup to avoid reloading on every request.
model = None
label_encoder = None

@app.on_event("startup")
def load_artifacts():
    """Load model and encoder when the API starts."""
    global model, label_encoder
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    encoder_path = os.path.join(MODELS_PATH, ENCODER_NAME)

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise RuntimeError("Model or encoder not found. Please train the model first.")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading label encoder from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("API is ready to accept requests.")


# --- Pydantic Models for Data Validation ---
class LandmarkSequence(BaseModel):
    """Defines the expected input data structure for a prediction request."""
    data: List[List[float]]


# --- API Endpoints ---
@app.get("/", summary="Root endpoint to check API status")
def read_root():
    """A simple endpoint to confirm that the API is running."""
    return {"status": "Sign Language Recognition API is running."}


@app.post("/predict", summary="Predict a sign from a sequence of landmarks")
def predict_sign(sequence: LandmarkSequence):
    """
    Accepts a sequence of landmarks, performs a prediction, and returns
    the predicted sign and confidence score.
    """
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # 1. Validate input data shape
        landmarks = np.array(sequence.data)
        if landmarks.shape[0] != FRAMES_PER_SEQUENCE:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence length. Expected {FRAMES_PER_SEQUENCE} frames, but got {landmarks.shape[0]}."
            )

        # 2. Reshape for model prediction
        X = np.expand_dims(landmarks, axis=0) # Add batch dimension -> (1, 45, num_features)
        
        # 3. Make prediction
        prediction = model.predict(X, verbose=0)[0]
        
        # 4. Decode prediction
        pred_index = np.argmax(prediction)
        confidence = float(prediction[pred_index])
        predicted_sign = label_encoder.inverse_transform([pred_index])[0]

        return {
            "predicted_sign": predicted_sign,
            "confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        # Catch potential errors during processing
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")