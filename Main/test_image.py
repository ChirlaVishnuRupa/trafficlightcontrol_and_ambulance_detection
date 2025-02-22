from ultralytics import YOLO
import cv2
import numpy as np

# Load both YOLO models
model_best = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\best.pt")  
model_yolo11n = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\yolo11n.pt")  

# Load an image
image_path = "C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\Data\\fireengine.png"
image = cv2.imread(image_path)

# Run both models
results_best = model_best(image, imgsz=1024, half=False)
results_yolo11n = model_yolo11n(image, imgsz=1024, half=False)

# Get detections from both models
detections_best = results_best[0].boxes.data.cpu().numpy()  
detections_yolo11n = results_yolo11n[0].boxes.data.cpu().numpy()  

# Object names
names_best = results_best[0].names
names_yolo11n = results_yolo11n[0].names

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

# Final list of detections
final_detections = []

# Track ambulances separately to remove duplicate labels
ambulance_boxes = []

# Process best.pt (keep only ambulances)
for det in detections_best:
    x1, y1, x2, y2, conf, cls = det
    label = names_best[int(cls)]
    if label == "ambulance":  
        ambulance_boxes.append((x1, y1, x2, y2))
        final_detections.append((x1, y1, x2, y2, conf, label))

# Process yolo11n.pt (keep all except ambulances and remove trucks overlapping ambulances)
for det in detections_yolo11n:
    x1, y1, x2, y2, conf, cls = det
    label = names_yolo11n[int(cls)]

    # If it's a truck and overlaps an ambulance, ignore it
    if label == "truck" and any(iou((x1, y1, x2, y2), ab) > 0.5 for ab in ambulance_boxes):
        continue  

    # Ignore ambulances detected by yolo11n.pt
    if label != "ambulance":
        final_detections.append((x1, y1, x2, y2, conf, label))

# Draw final detections
for x1, y1, x2, y2, conf, label in final_detections:
    color = (0, 0, 255) if label == "ambulance" else (0, 255, 0)  
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show cleaned image
cv2.imshow("YOLOv8 Fixed Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()