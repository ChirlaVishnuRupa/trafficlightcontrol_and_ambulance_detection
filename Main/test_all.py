from ultralytics import YOLO
import cv2
import numpy as np

# Load all three YOLO models
model_best = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\best.pt")  
model_yolo11n = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\yolo11n.pt")  
model_fire_engine = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\fire_engineyolo.pt")

# Load an image
image_path = "C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\Data\\amb_fire.png"
image = cv2.imread(image_path)

# Resize image to a smaller size
image = cv2.resize(image, (800, 600))  # Resize to 800x600

# Run all models
results_best = model_best(image, imgsz=640, half=False)
results_yolo11n = model_yolo11n(image, imgsz=640, half=False)
results_fire_engine = model_fire_engine(image, imgsz=640, half=False)

# Get detections from all models
detections_best = results_best[0].boxes.data.cpu().numpy()
detections_yolo11n = results_yolo11n[0].boxes.data.cpu().numpy()
detections_fire_engine = results_fire_engine[0].boxes.data.cpu().numpy()

# Object names
names_best = results_best[0].names
names_yolo11n = results_yolo11n[0].names
names_fire_engine = results_fire_engine[0].names

# IoU function to check box overlap
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Final list of detections
final_detections = []

# Track overlapping boxes
all_detections = []

def process_detections(detections, names, source):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = names[int(cls)]
        all_detections.append((x1, y1, x2, y2, conf, label, source))

process_detections(detections_best, names_best, "best")
process_detections(detections_fire_engine, names_fire_engine, "fire_engine")
process_detections(detections_yolo11n, names_yolo11n, "yolo11n")

# Remove overlapping detections (keep highest confidence)
for i, det1 in enumerate(all_detections):
    keep = True
    for j, det2 in enumerate(all_detections):
        if i != j and iou(det1[:4], det2[:4]) > 0.5:
            if det1[4] < det2[4]:  # Compare confidence scores
                keep = False
                break
    if keep:
        final_detections.append(det1)

# Correct label logic for fire engines
for i, det in enumerate(final_detections):
    x1, y1, x2, y2, conf, label, source = det
    
    # Fix truck label if fire engine detected
    if label == "truck":
        for fire_det in detections_fire_engine:
            if iou(det[:4], fire_det[:4]) > 0.5:
                label = "fire_engine"
                conf = max(conf, fire_det[4])  # Take the higher confidence
                source = "fire_engine"
                break

    # Update the final detection
    final_detections[i] = (x1, y1, x2, y2, conf, label, source)

# Draw final detections
for x1, y1, x2, y2, conf, label, source in final_detections:
    color = (0, 0, 255) if label == "ambulance" else (0, 140, 255) if label == "fire_engine" else (0, 255, 0)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the final result
cv2.imshow("YOLOv11 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Let me know if this works better! 🚀