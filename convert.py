from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")  # Replace with your YOLO model path

# Export to ONNX
model.export(format="onnx")