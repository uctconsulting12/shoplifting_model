from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO(r"E:\All_models\ppe_model\src\local_models\ppe_code\best.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'