from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11s.onnx'