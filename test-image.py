import argparse
import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import json

class Detect:
    def __init__(self, onnx_model, confidence_thres, iou_thres, mqtt_broker, mqtt_topic):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = ["Helm", "Mobil", "Motor", "No-Helm", "Pejalan-Kaki", "Pengendara-Motor"]
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # MQTT setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker)
        self.mqtt_topic = mqtt_topic

    def preprocess(self, frame):
        self.img_height, self.img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        detections = []
        x_factor, y_factor = self.img_width / 640, self.img_height / 640

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
                detections.append({
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "score": float(max_score),
                    "box": [left, top, width, height]
                })

        return detections

    def publish_detections(self, detections):
        message = [{"id": det["class_id"], "name": det["class_name"], "score": det["score"]} for det in detections]
        self.mqtt_client.publish(self.mqtt_topic, json.dumps(message))

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image from {image_path}")
            return

        img_data = self.preprocess(frame)
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: img_data})
        detections = self.postprocess(outputs)
        self.publish_detections(detections)
        print(f"Detections sent via MQTT: {detections}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.onnx", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--mqtt-broker", type=str, default="localhost", help="MQTT broker address.")
    parser.add_argument("--mqtt-topic", type=str, default="deteksi/objek", help="MQTT topic to publish detections.")
    args = parser.parse_args()

    detection = Detect(args.model, args.conf_thres, args.iou_thres, args.mqtt_broker, args.mqtt_topic)
    detection.process_image(args.image)