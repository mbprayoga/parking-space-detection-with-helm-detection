import cv2
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import logging

class ParkingDetectionModel:
    def __init__(self):
        logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
        
        self.model = YOLO('yolov8s.pt')
        self.video_path = 'test-park.mp4'
        self.frame_size = (1020, 500)
        self.frame_skip = 3

        self.cap = cv2.VideoCapture(self.video_path)
        self.polylines = []
        self.area_names = []
        self.class_list = []

        self.parking_status = {}

        self._load_files()

    def _load_files(self):
        """Load pickle file for parking segmentation and COCO class names."""
        with open('parkingsegment', 'rb') as f:
            data = pickle.load(f)
            self.polylines = data['polylines']
            self.area_names = data['area_names']

        with open('coco.txt', 'r') as f:
            self.class_list = f.read().split('\n')

        for area_name in self.area_names:
            self.parking_status[area_name] = "Empty"

    def open_video(self):
        """Open the video file for reading."""
        self.cap = cv2.VideoCapture(self.video_path)

    def reset_video(self):
        """Reset the video to the first frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_frame(self, frame):
        """Process a video frame and update parking area statuses."""
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for i, polyline in enumerate(self.polylines):
            is_filled = False
            for cx, cy in motorcycles_positions:
                if cv2.pointPolygonTest(polyline, (cx, cy), False) >= 0:
                    is_filled = True
                    break

            status = "Filled" if is_filled else "Empty"
            self.parking_status[self.area_names[i]] = status

            color = (0, 0, 255) if is_filled else (0, 255, 0)
            cv2.polylines(frame, [polyline], True, color, 2)
            cvzone.putTextRect(frame, f'{self.area_names[i]}', tuple(polyline[0]), 1, 1)

        return frame

    def run(self):
        """Read a video frame, process it, and return the frame and parking statuses."""
        ret, frame = self.cap.read()
        if not ret:
            self.reset_video()
            return None, self.parking_status

        processed_frame = self.process_frame(frame)
        return processed_frame, self.parking_status
