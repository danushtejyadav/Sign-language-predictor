# Sign Language Recognition MLOps Project

This project is a refactored version of a sign language recognition system, designed with MLOps principles in mind. The application is decoupled into three main parts: data collection, model training, and real-time inference.

## Project Structure

- **/data/raw/**: Stores the raw landmark data captured for each sign, organized in subfolders.
- **/models/**: Stores the trained model (`.h5` file) and the label encoder (`.pkl` file).
- **/src/**: Contains the core Python scripts for the ML pipeline.
  - `data_collection.py`: A command-line script to capture sign language data.
  - `train.py`: A command-line script to train the model on the collected data.
  - `utils.py`: Helper functions used by other scripts.
- **/app/**: Contains the client application for real-time sign detection.
  - `client.py`: A Tkinter GUI application for inference.
- `requirements.txt`: A list of required Python packages.

