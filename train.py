from ultralytics import YOLO
import json
import os

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# Load configuration
config = load_config()
DATASET_YAML = config['paths']['dataset_yaml']

# Initialize the model
model = YOLO("yolov5nu.pt")

# Train the model using your custom dataset
results = model.train(
    data=DATASET_YAML,  # Path to your dataset YAML file
    epochs=100,         # Number of training epochs
    imgsz=640,         # Image size
    batch=64,          # Batch size
    name='yolov5_custom'  # Name for the experiment
)
