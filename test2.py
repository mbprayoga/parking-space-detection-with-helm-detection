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
VIDEO_PATH = 'test-park.mp4'
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

# Indicators for parking areas
parking_status = {}  # Dictionary to store status for each parking area
area_labels = {}  # Store labels for updating status dynamically


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

    # Initialize parking area status
    for area_name in area_names:
        parking_status[area_name] = "Empty"
        area_labels[area_name] = ctk.CTkLabel(
            right_frame, text=f"{area_name}: Empty", fg_color="green", width=200, height=30, corner_radius=8
        )
        area_labels[area_name].pack(pady=5, padx=10)


def open_video():
    """Open the video file."""
    global cap
    cap = cv2.VideoCapture(VIDEO_PATH)


def process_frame(frame, count):
    """
    Process each frame for motorcycle detection and check if the motorcycle is inside the parking area.
    
    Args:
        frame (np.array): The current video frame.
        count (int): The current frame count.
    """
    if count % FRAME_SKIP != 0:
        return None  # Skip processing

    frame = cv2.resize(frame, FRAME_SIZE)
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")
    motorcycles_positions = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id]

        if 'motorcycle' in class_name:
            motorcycles_positions.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Update parking area status
    for i, polyline in enumerate(polylines):
        is_filled = False
        for cx, cy in motorcycles_positions:
            if cv2.pointPolygonTest(polyline, (cx, cy), False) >= 0:
                is_filled = True
                break

        # Update the label and status
        if is_filled:
            parking_status[area_names[i]] = "Filled"
            area_labels[area_names[i]].configure(
                text=f"{area_names[i]}: Filled", fg_color="red"
            )
        else:
            parking_status[area_names[i]] = "Empty"
            area_labels[area_names[i]].configure(
                text=f"{area_names[i]}: Empty", fg_color="green"
            )

        # Draw polyline on the frame
        color = (0, 0, 255) if is_filled else (0, 255, 0)
        cv2.polylines(frame, [polyline], True, color, 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0]), 1, 1)

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
