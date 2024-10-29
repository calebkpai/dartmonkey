import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import pygame
import numpy as np
import time
import torch

path_to_model = r'C:\Users\siuan\OneDrive\Documents\ML_Images'

model = YOLO(path_to_model, 'v8')               # Load the YOLOv8 model

cap = cv2.VideoCapture(0)                                   # Start video capture from webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")                  #Print Error message if webcam can't be opened
    exit()

while True:
   
    ret, frame = cap.read()                                 # Read frame from the webcam

    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame)                                  # Perform YOLOv8 detection on the frame
    annotated_frame = results[0].plot()                     # Draw the detection results on the frame
    cv2.imshow('Balloon Detection Lmao', annotated_frame)   # Show the frame with detection results

    if cv2.waitKey(1) & 0xFF == ord('q'):                   # Break the loop if the user presses q
        break

cap.release()                                               # Release resources
cv2.destroyAllWindows()