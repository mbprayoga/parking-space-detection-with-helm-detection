import argparse
import cv2
import numpy as np
import onnxruntime as ort

class HelmetDetection:
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names
        self.classes = ["Helm", "Mobil", "Motor", "No-Helm", "Pejalan-Kaki", "Pengendara-Motor"]

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

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(frame, box, score, class_id)

        return frame

    def run(self):
        """Run helmet detection on the provided video."""
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        cap = cv2.VideoCapture('test-helm.mp4')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_data = self.preprocess(frame)
            outputs = session.run(None, {session.get_inputs()[0].name: img_data})
            output_frame = self.postprocess(frame, outputs)
            return output_frame

        cap.release()
