import tkinter as tk
from tkinter import messagebox
import os
import ast
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import cv2

# Paths
DATA_FOLDER = "Data/"
AREAS_FOLDER = "Data/areas/"
IMAGE_FILES = [f"{DATA_FOLDER}1.jpg", f"{DATA_FOLDER}2.jpg", f"{DATA_FOLDER}3.jpg", f"{DATA_FOLDER}4.jpg"]

os.makedirs(AREAS_FOLDER, exist_ok=True)

# Global variables
selected_regions = [[] for _ in range(4)]
images = [None] * 4  
original_images = [None] * 4  # Store original images for red dot updates
current_image_index = 0  
selection_mode = False  
selection_done = False  

# Initialize GUI
root = ctk.CTk()
root.title("Traffic Control System")
root.geometry("700x500")
root.configure(bg="black")

# Function to load images
def load_images():
    global images, original_images
    for i, path in enumerate(IMAGE_FILES):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 300))  
        original_images[i] = Image.fromarray(img)
        images[i] = ImageTk.PhotoImage(original_images[i])

# Start area selection
def start_area_selection():
    global selection_mode, current_image_index
    selection_mode = True
    current_image_index = 0
    show_next_image()

# Show next image for selection
def show_next_image():
    global current_image_index
    if current_image_index >= 4:
        save_selected_areas()
        global selection_done
        selection_done = True
        root.destroy()  # Close GUI after selection
        return

    update_canvas()
    canvas.bind("<Button-1>", on_canvas_click)
    selected_regions[current_image_index] = []  
    info_label.configure(text=f"Select 4 points on Image {current_image_index + 1}")

# Handle mouse clicks for selection
def on_canvas_click(event):
    global current_image_index
    x, y = event.x, event.y
    selected_regions[current_image_index].append((x, y))

    draw_red_dot(x, y)
    
    if len(selected_regions[current_image_index]) == 4:  
        current_image_index += 1
        show_next_image()

# Draw red dot on selected point
def draw_red_dot(x, y):
    global current_image_index
    img = original_images[current_image_index].copy()
    draw = ImageDraw.Draw(img)
    draw.ellipse((x-3, y-3, x+3, y+3), fill="red", outline="red")  # Draw red dot
    images[current_image_index] = ImageTk.PhotoImage(img)
    update_canvas()

# Update the canvas with the current image
def update_canvas():
    canvas.create_image(0, 0, anchor=tk.NW, image=images[current_image_index])

# Save selected areas
def save_selected_areas():
    for i in range(4):
        file_path = os.path.join(AREAS_FOLDER, f"area{i+1}.txt")
        with open(file_path, "w") as f:
            f.write(str(selected_regions[i]))  

# Load selected areas from files
def load_selected_areas():
    for i in range(4):
        file_path = os.path.join(AREAS_FOLDER, f"area{i+1}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                selected_regions[i] = ast.literal_eval(f.read())  
        else:
            messagebox.showerror("Error", f"Missing area file: {file_path}")
            return
    global selection_done
    selection_done = True
    root.destroy()  # Close GUI after loading

# UI Elements
canvas = tk.Canvas(root, width=500, height=300, bg="black")
canvas.pack()

info_label = ctk.CTkLabel(root, text="Select Old or New Areas", font=("Arial", 16))
info_label.pack()

btn_new = ctk.CTkButton(root, text="New", command=start_area_selection)
btn_new.pack(pady=10)

btn_old = ctk.CTkButton(root, text="Old", command=load_selected_areas)
btn_old.pack(pady=10)

load_images()
root.mainloop()
