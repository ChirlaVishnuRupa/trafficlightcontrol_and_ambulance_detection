from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

# Load YOLO models
model_best = YOLO("model/best.pt")  
model_yolo11n = YOLO("model/yolo11n.pt")  

# Paths
DATA_FOLDER = "Data/"
AREAS_FOLDER = "Data/areas/"
IMAGE_FILES = [f"{DATA_FOLDER}1.jpg", f"{DATA_FOLDER}2.jpg", f"{DATA_FOLDER}3.jpg", f"{DATA_FOLDER}4.jpg"]

os.makedirs(AREAS_FOLDER, exist_ok=True)  

# IoU function to check box overlap
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

# Function to detect vehicles using both YOLO models
def detect_vehicles():
    for i, path in enumerate(IMAGE_FILES):
        img = cv2.imread(path)

        # Run both models
        results_best = model_best(img)
        results_yolo11n = model_yolo11n(img)

        # Get detections
        detections_best = results_best[0].boxes.data.cpu().numpy()
        detections_yolo11n = results_yolo11n[0].boxes.data.cpu().numpy()

        names_best = results_best[0].names
        names_yolo11n = results_yolo11n[0].names

        final_detections = []
        ambulance_boxes = []

        # Process detections from `best.pt` (Keep only ambulances)
        for det in detections_best:
            x1, y1, x2, y2, conf, cls = det
            label = names_best[int(cls)]
            if label == "ambulance":  
                ambulance_boxes.append((x1, y1, x2, y2))
                final_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), label])  

        # Process detections from `yolo11n.pt` (Keep all except ambulances & remove overlapping trucks)
        for det in detections_yolo11n:
            x1, y1, x2, y2, conf, cls = det
            label = names_yolo11n[int(cls)]

            # If truck overlaps an ambulance, ignore it
            if label == "truck" and any(iou((x1, y1, x2, y2), ab) > 0.5 for ab in ambulance_boxes):
                continue  
            
            # Ignore ambulances detected by `yolo11n.pt`
            if label != "ambulance":
                final_detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), label])  

        # Save detections as JSON
        detection_path = os.path.join(AREAS_FOLDER, f"detections_{i+1}.txt")
        with open(detection_path, "w") as f:
            json.dump(final_detections, f)

# Function to show detected frames with bounding boxes
def show_resultant_frames():
    for i, path in enumerate(IMAGE_FILES):
        img = cv2.imread(path)

        results_best = model_best(img)
        results_yolo11n = model_yolo11n(img)

        # Get detections from both models
        detections_best = results_best[0].boxes.data.cpu().numpy()
        detections_yolo11n = results_yolo11n[0].boxes.data.cpu().numpy()

        names_best = results_best[0].names
        names_yolo11n = results_yolo11n[0].names

        final_detections = []
        ambulance_boxes = []

        for det in detections_best:
            x1, y1, x2, y2, conf, cls = det
            label = names_best[int(cls)]
            if label == "ambulance":  
                ambulance_boxes.append((x1, y1, x2, y2))
                final_detections.append((x1, y1, x2, y2, conf, label))

        for det in detections_yolo11n:
            x1, y1, x2, y2, conf, cls = det
            label = names_yolo11n[int(cls)]

            if label == "truck" and any(iou((x1, y1, x2, y2), ab) > 0.5 for ab in ambulance_boxes):
                continue  
            if label != "ambulance":
                final_detections.append((x1, y1, x2, y2, conf, label))

        # Draw bounding boxes on the image
        for x1, y1, x2, y2, conf, label in final_detections:
            color = (0, 0, 255) if label == "ambulance" else (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(f"Resultant Frame {i+1}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run vehicle detection
detect_vehicles()
