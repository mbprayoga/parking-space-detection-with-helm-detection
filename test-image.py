import argparse
import cv2
import numpy as np
import onnxruntime as ort
import serial
import os
from werkzeug.utils import secure_filename

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

    def preprocess(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image_data = img / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def postprocess(self, output):
        outputs = np.squeeze(output[0]) 
        x_factor, y_factor = self.img_width / 640, self.img_height / 640

        detections = []
        for detection in outputs.T:
            x, y, w, h = detection[:4]
            classes_scores = detection[4:]
            max_score = np.amax(classes_scores)
            class_id = np.argmax(classes_scores)
            if max_score >= self.confidence_thres:
                detections.append({
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "score": float(max_score),
                    "box": [
                        int((x - w / 2) * x_factor),
                        int((y - h / 2) * y_factor),
                        int(w * x_factor),
                        int(h * y_factor)
                    ]
                })

        if len(detections) > 0:
            final_detections = []
            for class_id in range(len(self.classes)):
                class_detections = [det for det in detections if det["class_id"] == class_id]
                if class_detections:
                    boxes = np.array([det["box"] for det in class_detections])
                    scores = np.array([det["score"] for det in class_detections])
                    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_thres, self.iou_thres)
                    final_detections.extend([class_detections[i] for i in indices.flatten()])
            detections = final_detections

        return detections

    def send_detections_serial(self, detections):
        message = [
            {
                "id": int(det["class_id"]),
                "name": det["class_name"],
                "score": det["score"],
                "box": det["box"]
            }
            for det in detections
        ]
        try:
            if self.serial_connection:
                send_value = '0'
                detected_classes = {det["class_name"] for det in detections}
                if "unhelmeted" in detected_classes:
                    send_value = '0'
                elif "helmeted" in detected_classes and "biker" in detected_classes:
                    send_value = '1'
                self.serial_connection.write(send_value.encode('utf-8') + b'\n')
                print(f"Value sent: {send_value}")
        except Exception as e:
            print(f"Error sending data over serial: {e}")

    def draw_bounding_boxes(self, frame, detections):
        thickness = 10
        font_scale = 2.5
        font_thickness = 3

        for det in detections:
            box = det["box"]
            class_name = det["class_name"]
            score = det["score"]
            color = self.color_palette[class_name]

            x1 = box[0]
            y1 = box[1]
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    def process_image(self, image_path):
        if not os.path.isfile(image_path):
            print(f"Error: File {image_path} does not exist.")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image from {image_path}")
            return

        img_data = self.preprocess(frame)
        try:
            session = ort.InferenceSession(
                self.onnx_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            outputs = session.run(None, {session.get_inputs()[0].name: img_data})
            
            # print(f"Output shapes: {[output.shape for output in outputs]}")
            
            detections = self.postprocess(outputs)
            self.send_detections_serial(detections)

            self.draw_bounding_boxes(frame, detections)

            output_image_path = os.path.join("detect/image", secure_filename(os.path.basename(image_path)))
            cv2.imwrite(output_image_path, frame)
            # print(f"Processed image saved to {output_image_path}")

        except Exception as e:
            print(f"Error during model inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolov8/v1/best.onnx", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--serial-port", type=str, default="COM8", help="Serial port for communication.")
    parser.add_argument("--serial-baudrate", type=int, default=115200, help="Serial baud rate.")
    args = parser.parse_args()

    detection = Detect(args.model, args.conf_thres, args.iou_thres, args.serial_port, args.serial_baudrate)
    detection.process_image(args.image)