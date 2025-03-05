import time
import os
import json
import customtkinter as ctk
import cv2
import numpy as np
from vehicle_detection import model_best, model_yolo11n, IMAGE_FILES, iou

# Paths
AREAS_FOLDER = "Data/areas/"
IMAGE_COUNT = 4  

# Initialize GUI
root = ctk.CTk()
root.title("Smart Traffic Control System")
root.geometry("700x500")
root.configure(bg="black")

# Header
header = ctk.CTkLabel(root, text="Smart Traffic Control System", font=("Arial", 24, "bold"), fg_color="black", text_color="white")
header.pack(pady=10)

# Grid Layout for Traffic Signals
frame_container = ctk.CTkFrame(root, fg_color="black")
frame_container.pack(pady=10)

frame_labels = []
for i in range(IMAGE_COUNT):
    frame = ctk.CTkFrame(frame_container, fg_color="black", width=200, height=150, corner_radius=10)
    frame.grid(row=i//2, column=i%2, padx=20, pady=20)
    
    label = ctk.CTkLabel(frame, text=f"STOP\nVehicles: 0\nTime: --s", font=("Arial", 18, "bold"), fg_color="red", text_color="white", width=200, height=100)
    label.pack(pady=10)
    frame_labels.append(label)

# Function to read detections
def read_detections():
    vehicle_counts = [0] * IMAGE_COUNT
    ambulance_detected = [False] * IMAGE_COUNT
    for i in range(IMAGE_COUNT):
        detection_path = os.path.join(AREAS_FOLDER, f"detections_{i+1}.txt")
        if os.path.exists(detection_path):
            with open(detection_path, "r") as f:
                try:
                    detections = json.load(f)  
                except json.JSONDecodeError:
                    continue
            count = len(detections)
            for _, _, _, _, _, label in detections:
                if label == "ambulance":
                    ambulance_detected[i] = True  
            vehicle_counts[i] = count
    return vehicle_counts, ambulance_detected

# Function to update GUI labels
def update_display(index, status, time_allocated, vehicle_counts):
    if 0 <= index < len(frame_labels):
        color = "green" if status == "GO" or status == "AMBULANCE" else "red"
        frame_labels[index].configure(text=f"{status}\nVehicles: {vehicle_counts[index]}\nTime: {time_allocated}s", fg_color=color)
        root.update()


def start_traffic_simulation():
    while True:
        vehicle_counts, ambulance_detected = read_detections()
        
        # Find all ambulance frames
        ambulance_frames = [i for i in range(IMAGE_COUNT) if ambulance_detected[i]]

        if ambulance_frames:
            for frame in ambulance_frames:
                # Stop all frames first
                for j in range(IMAGE_COUNT):
                    update_display(j, "STOP", 0, vehicle_counts)
                
                # Allocate time for the ambulance frame
                ambulance_time = max(5, min(vehicle_counts[frame] * 2, 30))
                update_display(frame, "AMBULANCE", ambulance_time, vehicle_counts)

                # Countdown for the ambulance frame
                for t in range(ambulance_time, 0, -1):
                    frame_labels[frame].configure(
                        text=f"AMBULANCE\nVehicles: {vehicle_counts[frame]}\nTime: {t}s",
                        fg_color="green"
                    )
                    root.update()
                    time.sleep(1)
                
                # Stop the ambulance frame after processing
                update_display(frame, "STOP", 0, vehicle_counts)
        
        # Regular cycle for non-ambulance frames
        for i in range(IMAGE_COUNT):
            if i in ambulance_frames:
                continue  # Skip frames that already processed ambulances
            
            time_allocated = max(5, min(vehicle_counts[i] * 2, 30))
            update_display(i, "GO", time_allocated, vehicle_counts)

            # Countdown for the frame
            for t in range(time_allocated, 0, -1):
                frame_labels[i].configure(
                    text=f"GO\nVehicles: {vehicle_counts[i]}\nTime: {t}s",
                    fg_color="green"
                )
                root.update()
                time.sleep(1)
            
            update_display(i, "STOP", 0, vehicle_counts)


# Function to show resultant frames
def show_resultant_frames():
    for i, path in enumerate(IMAGE_FILES):
        img = cv2.imread(path)
        results_best = model_best(img)
        results_yolo11n = model_yolo11n(img)
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
                final_detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), label))
        for det in detections_yolo11n:
            x1, y1, x2, y2, conf, cls = det
            label = names_yolo11n[int(cls)]
            if label == "truck" and any(iou((x1, y1, x2, y2), ab) > 0.5 for ab in ambulance_boxes):
                continue  
            if label != "ambulance":
                final_detections.append((float(x1), float(y1), float(x2), float(y2), float(conf), label))
        for x1, y1, x2, y2, conf, label in final_detections:
            color = (0, 0, 255) if label == "ambulance" else (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow(f"Resultant Frame {i+1}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Control Panel
btn_show_frames = ctk.CTkButton(root, text="Show Resultant Frames", command=show_resultant_frames)
btn_show_frames.pack(pady=10)

# Start simulation
start_traffic_simulation()
root.mainloop()
