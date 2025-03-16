import cv2
import customtkinter as ctk
from ultralytics import YOLO
import numpy as np

# Load YOLO models
model_best = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\best.pt")
model_yolo11n = YOLO("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\model\\yolo11n.pt")

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

# Detection function
def detect_vehicles(image_path):
    image = cv2.imread(image_path)
    results_best = model_best(image, imgsz=1024, half=False)
    results_yolo11n = model_yolo11n(image, imgsz=1024, half=False)

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

    for x1, y1, x2, y2, conf, label in final_detections:
        color = (0, 0, 255) if label == "ambulance" else (0, 255, 0)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GUI
def run_demo():
    app = ctk.CTk()
    app.geometry("500x300")
    app.title("Rainy Condition Detection Demo")

    ctk.CTkLabel(app, text="check how model detects in rainy conditions", font=("Arial", 20)).pack(pady=20)

    def detect_normal():
        detect_vehicles("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\Data\\rain.png")

    def detect_ambulance():
        detect_vehicles("C:\\Users\\hp\\OneDrive\\Desktop\\traffic control and ambulance detection\\Data\\amb_rain.jpg")

    ctk.CTkButton(app, text="Normal Vehicles", command=detect_normal).pack(pady=10)
    ctk.CTkButton(app, text="Ambulance", command=detect_ambulance).pack(pady=10)

    app.mainloop()

if __name__ == "__main__":
    run_demo()
