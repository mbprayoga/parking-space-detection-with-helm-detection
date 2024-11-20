import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import customtkinter as ctk
from PIL import Image, ImageTk

# Global variables
MODEL_PATH = 'yolov8s.pt'
VIDEO_PATH = 'easy1.mp4'
PICKLE_FILE = 'parkingsegment'
COCO_CLASSES_FILE = 'coco.txt'
FRAME_SIZE = (1020, 500)
FRAME_SKIP = 3

model = None
cap = None
polylines = []
area_names = []
class_list = []

# Initialize CustomTkinter
app = ctk.CTk()
app.title("Parking Detection System")
app.state('zoomed')  # Fullscreen

# Frames
left_frame = ctk.CTkFrame(app, width=800, corner_radius=10)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = ctk.CTkFrame(app, width=400, corner_radius=10)
right_frame.pack(side="right", fill="both", expand=True)

video_label = ctk.CTkLabel(left_frame, text="", anchor="center")
video_label.pack(fill="both", expand=True)


def initialize_model():
    """Initialize the YOLO model."""
    global model
    model = YOLO(MODEL_PATH)


def load_files():
    """Load the necessary files: pickle file for parking segmentation and class names."""
    global polylines, area_names, class_list
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
    
    with open(COCO_CLASSES_FILE, "r") as f:
        class_list = f.read().split("\n")


def open_video():
    """Open the video file."""
    global cap
    cap = cv2.VideoCapture(VIDEO_PATH)


def process_frame(frame, count):
    """
    Process each frame for car detection and check if the car is inside the parking area.
    
    Args:
        frame (np.array): The current video frame.
        count (int): The current frame count.
    """
    if count % FRAME_SKIP != 0:
        return None  # Skip processing

    frame = cv2.resize(frame, FRAME_SIZE)
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")
    cars_positions = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id]

        if 'car' in class_name:
            cars_positions.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

        for cx, cy in cars_positions:
            if cv2.pointPolygonTest(polyline, (cx, cy), False) >= 0:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)

    return frame


def display_video():
    """Display the video on the left frame."""
    global cap
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        processed_frame = process_frame(frame, count)
        if processed_frame is not None:
            # Convert frame to RGB for displaying in Tkinter
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        app.update_idletasks()
        app.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def main():
    """Main function to execute the parking detection pipeline."""
    initialize_model()
    load_files()
    open_video()
    display_video()


if __name__ == "__main__":
    main()
