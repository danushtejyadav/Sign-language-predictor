import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import requests
import json

# Adjust import path to access src utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import holistic, extract_landmarks

class RecognitionClient:
    def __init__(self):
        # --- API Configuration ---
        self.api_url = "http://localhost:8000/predict"
        
        # --- Recognition Parameters ---
        self.sequence_length = 3
        self.fps = 15
        self.frames_per_sequence = self.sequence_length * self.fps
        self.sequence_buffer = []
        self.no_detection_counter = 0
        self.detection_threshold = 15

        # --- UI Components ---
        self.root = None
        self.is_recognizing = False
        
        # --- Threading ---
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # --- Initialization ---
        self.setup_ui()
        
    def setup_ui(self):
        """Creates the user interface."""
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition Client")
        self.root.geometry("800x600")
        
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        recognition_frame = ttk.LabelFrame(main_frame, text="Recognition")
        recognition_frame.pack(fill=tk.X, pady=10)
        
        self.recognize_btn = ttk.Button(recognition_frame, text="Start Recognition", command=self.toggle_recognition)
        self.recognize_btn.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(recognition_frame, text="Detected Sign:").pack(anchor=tk.W, padx=5, pady=5)
        self.detected_sign_var = tk.StringVar(value="--")
        sign_label = ttk.Label(recognition_frame, textvariable=self.detected_sign_var, font=("Arial", 20, "bold"))
        sign_label.pack(padx=5, pady=5)
        
        # --- Lines to re-add ---
        self.confidence_var = tk.DoubleVar(value=0)
        self.confidence_bar = ttk.Progressbar(recognition_frame, variable=self.confidence_var, maximum=100)
        self.confidence_bar.pack(fill=tk.X, padx=5, pady=10)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.root.destroy()
            return
            
        self.is_running = True
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        self.update_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def process_video(self):
        """Processes video frames and calls the API for prediction."""
        prev_time = 0
        while self.is_running:
            curr_time = time.time()
            if (curr_time - prev_time) < 1.0/self.fps:
                time.sleep(max(0, 1.0/self.fps - (curr_time - prev_time)))
            prev_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if self.is_recognizing:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                landmarks = extract_landmarks(results)
                has_landmarks = np.any(landmarks[:126] > 0.001)

                if has_landmarks:
                    self.sequence_buffer.append(landmarks)
                    self.no_detection_counter = 0
                    if len(self.sequence_buffer) > self.frames_per_sequence:
                        self.sequence_buffer.pop(0)

                    # --- API Call Section ---
                    if len(self.sequence_buffer) == self.frames_per_sequence:
                        try:
                            # Prepare data in the format expected by the API
                            payload = {
                                "data": [arr.tolist() for arr in self.sequence_buffer]
                            }
                            
                            # Make the HTTP POST request
                            response = requests.post(self.api_url, json=payload, timeout=0.5)
                            response.raise_for_status()  # Raise an exception for bad status codes
                            
                            api_result = response.json()
                            sign = api_result.get("predicted_sign")
                            confidence = float(api_result.get("confidence", 0.0)) * 100
                            
                            self.result_queue.put({'sign': sign, 'confidence': confidence})

                        except requests.exceptions.RequestException as e:
                            # Handle connection errors, timeouts, etc.
                            print(f"API Error: {e}")
                            self.result_queue.put({'sign': "API Error", 'confidence': 0})
                else:
                    self.no_detection_counter += 1
                
                if self.no_detection_counter > self.detection_threshold:
                    self.result_queue.put({'sign': "--", 'confidence': 0})
                    self.sequence_buffer.clear()
                    self.no_detection_counter = 0

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def update_ui(self):
        """Updates UI elements with the latest data."""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self._update_video_display(frame)
            
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.detected_sign_var.set(result.get('sign', '--'))
                self.confidence_var.set(result.get('confidence', 0))

        except queue.Empty:
            pass
        self.root.after(30, self.update_ui)

    def _update_video_display(self, frame):
        """Updates the video label with a new frame."""
        h, w = frame.shape[:2]
        scale = min(640/w, 480/h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def toggle_recognition(self):
        """Toggles recognition mode on and off."""
        if self.is_recognizing:
            self.is_recognizing = False
            self.recognize_btn.config(text="Start Recognition")
            self.detected_sign_var.set("--")
            self.confidence_var.set(0)
            self.sequence_buffer.clear()
        else:
            self.is_recognizing = True
            self.recognize_btn.config(text="Stop Recognition")

    def on_close(self):
        """Handles application closing."""
        self.is_running = False
        time.sleep(0.1) 
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    app = RecognitionClient()
    app.root.mainloop()

if __name__ == "__main__":
    main()