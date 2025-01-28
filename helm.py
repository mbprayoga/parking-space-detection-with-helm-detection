import argparse
import cv2
import numpy as np
import onnxruntime as ort
import threading
import serial

class HelmetDetection:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        
        self.serial_connection = None
        self.serial_port = "COM8"
        self.serial_baudrate = 115200
        try:
            self.serial_connection = serial.Serial(self.serial_port, self.serial_baudrate, timeout=1)
            print(f"Connected to serial port {serial_port} at {serial_baudrate} baud.")
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            exit(1)

        # Load the class names
        self.classes = ['biker', 'helmeted', 'person', 'unhelmeted']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def draw_detections(self, img, box, score, class_id):
        """Draws bounding boxes and labels on the input image based on the detected objects."""
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, frame):
        """Preprocess the frame before feeding it into the model."""
        self.img_height, self.img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, frame, output):
        """Postprocess the model outputs."""
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        x_factor = self.img_width / 640
        y_factor = self.img_height / 480

        detection_results = []
        
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                detection_results.append({"class_name": self.classes[class_id]})

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(frame, box, score, class_id)

        return frame, detection_results

    def send_detections_serial(self, detections):
        """Send the detected objects to a serial connection."""
        try:
            if self.serial_connection:
                detected_classes = {det["class_name"] for det in detections}
                if "person" in detected_classes or "biker" in detected_classes:
                    current_time = time.time()
                    if current_time - self.last_capture_time >= 5:
                        self.last_capture_time = current_time
                        self.serial_connection.write(b"1")
                        self.serial_connection.flush()
        except Exception as e:
            print(f"Error sending detections: {e}")
    
    def run(self, update_callback):
        """Run helmet detection on the provided video and call the update callback."""
        session = ort.InferenceSession(self.onnx_model, providers=["AzureExecutionProvider", "CPUExecutionProvider"])
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_data = self.preprocess(frame)
            outputs = session.run(None, {session.get_inputs()[0].name: img_data})
            output_frame, detection_results = self.postprocess(frame, outputs)
            update_callback(output_frame, detection_results)
            send_detections_serial(self, detection_results)
            
            

        cap.release()