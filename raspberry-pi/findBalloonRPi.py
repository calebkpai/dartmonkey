import cv2
import numpy as np
from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import subprocess

result = subprocess.run(['sudo', 'pigpiod'], check=True, text=True, capture_output=True)
print("Pigpiod initialized.")
path_to_model = "/home/user/Desktop/dart/best.pt"  # Path to the trained YOLO model
model = YOLO(path_to_model, 'v8n')                                   # Load YOLOv8 model

vid = cv2.VideoCapture("/dev/video0")                                           # Initialize video capture with lower resolution for faster processing
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #640
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #480
print("Video started")

pigpio_factory = PiGPIOFactory()                                    # Initialize servos with PiGPIO factory
horizontal_servo = AngularServo(13, pin_factory=pigpio_factory)
vertical_servo = AngularServo(12, pin_factory=pigpio_factory)
horizontal_servo.angle, vertical_servo.angle = 0, 0
sleep(2)
print("Servos initialized.")

Kp, Ki, Kd = 100, 0.1, 0.5                                          # PID Control constants 
cx, cy = -1, 1                                                      # Flip signs for servo direction correction
                                                           
alpha = 0.1                                                         # Low-pass filter coefficient for smoothing. Smaller is smoother

filtered_angle1 = horizontal_servo.angle                                      # Initial filtered angle for servo 1
filtered_angle2 = vertical_servo.angle                                      # Initial filtered angle for servo 2
integral_x, integral_y = 0, 0                                       # Initial integral terms for PID
previous_error_x, previous_error_y = 0, 0                           # Initial previous errors for PID
              
def getAdjustment(windowMax, x):                                    # Function to get adjustment direction and magnitude for PID control
    normalized_adjustment = x / windowMax - 0.5                     # Normalize the adjustment value
    adjustment_direction = -1 if normalized_adjustment > 0 else 1   # Determine direction
    return abs(round(normalized_adjustment, 1)), adjustment_direction  # Return magnitude and direction

while True:
    ret, frame = vid.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model(frame)
    annotated_frame = results[0].plot()                             # Draw the detection results on the frame

    if results[0].boxes:                                            # Apply PID control if an object is detected
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]                            # Get the center of the bounding box 
            center_x = int((x1 + x2) // 2)                          # Get the coordinates of the box
            center_y = int((y1 + y2) // 2)

            window = frame.shape                                    # Get frame size
            xmag, xdir = getAdjustment(window[1], center_x)         # Window width is for x-axis
            ymag, ydir = getAdjustment(window[0], center_y)         # Window height is for y-axis

            error_x = xdir * xmag                                   # PID terms for x and y
            error_y = ydir * ymag

            integral_x += error_x                                   # Integral accumulation
            integral_y += error_y

            derivative_x = error_x - previous_error_x               # Derivative terms
            derivative_y = error_y - previous_error_y

            adj_x = cx * (Kp * error_x + Ki * integral_x + Kd * derivative_x)# PID-based adjustments
            adj_y = cy * (Kp * error_y + Ki * integral_y + Kd * derivative_y)

            previous_error_x = error_x                              # Update previous errors
            previous_error_y = error_y

            target_angle1 = max(-90, min(90, horizontal_servo.angle + adj_x)) # Apply the low-pass filter to servo adjustments
            target_angle2 = max(-90, min(90, vertical_servo.angle + adj_y))
            filtered_angle1 = (alpha * target_angle1) + ((1 - alpha) * filtered_angle1)
            filtered_angle2 = (alpha * target_angle2) + ((1 - alpha) * filtered_angle2)

            horizontal_servo.angle = filtered_angle1                          # Update servos with filtered angles
            vertical_servo.angle = filtered_angle2

    cv2.imshow('PiCamera Vision', annotated_frame)    # Show the annotated frame with object detections
    #cv2.imshow("Grayscale Camera Feed")

    if cv2.waitKey(1) & 0xFF == ord('q'):                           # Break the loop if the user presses 'q'
        break

vid.release()                                                       # Release resources
cv2.destroyAllWindows()
