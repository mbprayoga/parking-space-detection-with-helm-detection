import argparse
import cv2
import numpy as np
import onnxruntime as ort
import serial
import threading
import time
import os

class Detect:
    def __init__(self, onnx_model, confidence_thres, iou_thres, serial_port, serial_baudrate):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = ['biker', 'helmeted', 'person', 'unhelmeted']
        self.color_palette = {
            'biker': (139, 0, 0),       # Biru tua
            'helmeted': (0, 255, 0),    # Hijau
            'person': (0, 255, 255),    # Kuning
            'unhelmeted': (0, 0, 255)   # Merah
        }

        self.serial_connection = None
        try:
            self.serial_connection = serial.Serial(serial_port, serial_baudrate, timeout=1)
            print(f"Connected to serial port {serial_port} at {serial_baudrate} baud.")
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            exit(1)

        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.warm_up_model()
        self.last_capture_time = 0

        # Ensure the capture directory exists
        self.capture_dir = "detect"
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)

    def warm_up_model(self):
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        self.session.run(None, {self.session.get_inputs()[0].name: dummy_input})
        print("Model warmed up.")

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[self.classes[class_id]]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x, label_y = x1, max(y1 - 10, label_height)
        overlay = img.copy()
        cv2.rectangle(overlay, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image_data = img / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def postprocess(self, frame, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        x_factor, y_factor = self.img_width / 640, self.img_height / 640
        detections = []
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
                detections.append({
                    "box": [left, top, width, height],
                    "score": max_score,
                    "class_id": class_id,
                    "class_name": self.classes[class_id]
                })
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        if len(indices) > 0:
            for i in indices.flatten():
                box, score, class_id = boxes[i], scores[i], class_ids[i]
                self.draw_detections(frame, box, score, class_id)
        return frame, detections

    def send_detections_serial(self, detections, frame):
        try:
            if self.serial_connection:
                detected_classes = {det["class_name"] for det in detections}
                if "person" in detected_classes or "biker" in detected_classes:
                    current_time = time.time()
                    if current_time - self.last_capture_time >= 5:
                        self.last_capture_time = current_time
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = os.path.join(self.capture_dir, f"capture_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"Image saved: {filename}")

                        # Send value to serial after saving the image
                        send_value = '1' if "unhelmeted" in detected_classes else '0'
                        self.serial_connection.write(send_value.encode('utf-8') + b'\n')
                        print(f"Value sent: {send_value}")

        except Exception as e:
            print(f"Error sending data over serial: {e}")

    def capture_and_process(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video capture source {source}.")
            return

        def process_frame():
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                img_data = self.preprocess(frame)
                try:
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
                    output_frame, detections = self.postprocess(frame, outputs)
                    self.send_detections_serial(detections, frame)
                    cv2.imshow('Detection', output_frame)

                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        break

                except Exception as e:
                    print(f"Error during model inference: {e}")

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=process_frame).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolov8/v3/best.onnx", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--serial-port", type=str, default="COM8", help="Serial port for communication.")
    parser.add_argument("--serial-baudrate", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--source", type=int, default=0, help="Source of the video feed (default is 0 for webcam).")
    args = parser.parse_args()

    detection = Detect(args.model, args.conf_thres, args.iou_thres, args.serial_port, args.serial_baudrate)
    detection.capture_and_process(args.source)