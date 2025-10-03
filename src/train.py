import os
import json
import numpy as np
import pickle
import mlflow  # MLflow integration
import mlflow.tensorflow  # MLflow integration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATA_PATH = "data/raw"
MODELS_PATH = "models"
SEQUENCE_LENGTH = 3  # Seconds
FPS = 15
FRAMES_PER_SEQUENCE = SEQUENCE_LENGTH * FPS

def train_model():
    """
    Loads data, trains the LSTM model, and saves the artifacts while tracking
    the experiment with MLflow.
    """
    os.makedirs(MODELS_PATH, exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    X, y = [], []
    sign_names = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

    if len(sign_names) < 2:
        print("Error: At least two different signs are required for training.")
        return

    for sign in sign_names:
        sign_path = os.path.join(DATA_PATH, sign)
        for sample_file in os.listdir(sign_path):
            if sample_file.endswith('.json'):
                with open(os.path.join(sign_path, sample_file), 'r') as f:
                    sequence = json.load(f)
                    if len(sequence) == FRAMES_PER_SEQUENCE:
                        X.append(sequence)
                        y.append(sign)

    if not X:
        print("Error: No valid data found for training.")
        return

    print(f"Loaded {len(X)} samples across {len(sign_names)} signs.")

    # 2. Preprocess Data
    print("Preprocessing data...")
    X = np.array(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    encoder_path = os.path.join(MODELS_PATH, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved locally to {encoder_path}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # --- MLflow Experiment Tracking ---
    mlflow.set_experiment("Sign Language Recognition")

    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_name}")
        
        # MLflow integration: Enable autologging for TensorFlow/Keras
        mlflow.tensorflow.autolog(log_models=True, disable=False)

        # MLflow integration: Log custom parameters
        mlflow.log_param("num_signs", len(sign_names))
        mlflow.log_param("total_samples", len(X))
        mlflow.log_param("frames_per_sequence", FRAMES_PER_SEQUENCE)

        # 3. Build Model
        print("Building LSTM model...")
        input_shape = (X.shape[1], X.shape[2])
        num_classes = len(label_encoder.classes_)
        
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.3),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        model.summary()

        # 4. Train Model
        print("Training model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=1
        )

        # 5. Save Model Locally (Optional, as MLflow also saves it)
        model_path = os.path.join(MODELS_PATH, 'sign_model.h5')
        model.save(model_path)
        print(f"Model saved locally to {model_path}")

        # MLflow integration: Log the label encoder as an artifact
        mlflow.log_artifact(encoder_path, artifact_path="label_encoder")
        
        val_accuracy = history.history['val_categorical_accuracy'][-1] * 100
        
        # MLflow integration: Log final summary metric
        mlflow.log_metric("final_val_accuracy", val_accuracy)

        print(f"\n--- Training Complete ---")
        print(f"Final Validation Accuracy: {val_accuracy:.2f}%")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Run `mlflow ui` to view the experiment results.")

if __name__ == "__main__":
    train_model()