import cv2
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import logging
import os
import threading

class ParkingDetectionModel:
    area_names = [] 

    def __init__(self, video_path, segmentation_path):
        logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
        
        self.model = YOLO('models/pre-trained/yolov8s.pt')
        self.video_path = video_path
        self.segmentation_path = segmentation_path
        self.frame_size = (1020, 500)

        self.cap = None
        self.polylines = []
        self.class_list = []
        self.parking_status = {}

        self._load_files()

    def _load_files(self):
        """Load pickle file for parking segmentation and COCO class names."""
        with open(self.segmentation_path, 'rb') as f:
            data = pickle.load(f)
            self.polylines = data['polylines']
            for area_name in data['area_names']:
                if area_name not in ParkingDetectionModel.area_names:
                    ParkingDetectionModel.area_names.append(area_name)

        with open('models/pre-trained/coco.txt', 'r') as f:
            self.class_list = f.read().split('\n')

        for area_name in ParkingDetectionModel.area_names:
            self.parking_status[area_name] = "Empty"

    def process_frame(self, frame):
        """Process a video frame and update parking area statuses."""
        if frame is None:
            return None  # Return None if the frame is invalid

        frame = cv2.resize(frame, self.frame_size)
        results = self.model.predict(frame)

        detections = pd.DataFrame(results[0].boxes.data).astype("float")
        motorcycles_positions = []

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            class_name = self.class_list[class_id]

            if 'motorcycle' in class_name:
                motorcycles_positions.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])

        for i, polyline in enumerate(self.polylines):
            is_filled = False
            for cx, cy in motorcycles_positions:
                if cv2.pointPolygonTest(polyline, (cx, cy), False) >= 0:
                    is_filled = True
                    break

            status = "Filled" if is_filled else "Empty"
            self.parking_status[ParkingDetectionModel.area_names[i]] = status

    def run(self, update_callback):
        """Read a video frame, process it, and call the update callback."""
        self.cap = cv2.VideoCapture(self.video_path)
        ret, frame = self.cap.read()

        self.process_frame(frame)
        update_callback(self.parking_status)
        self.cap.release()