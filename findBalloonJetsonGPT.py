import cv2
import numpy as np
import Jetson.GPIO as GPIO
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Pin configuration
HORIZONTAL_PIN = 13
VERTICAL_PIN = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setup(HORIZONTAL_PIN, GPIO.OUT)
GPIO.setup(VERTICAL_PIN, GPIO.OUT)

# Initialize PWM for servos
horizontal_servo = GPIO.PWM(HORIZONTAL_PIN, 50)  # 50 Hz frequency
vertical_servo = GPIO.PWM(VERTICAL_PIN, 50)
horizontal_servo.start(7.5)  # Neutral position (90 degrees)
vertical_servo.start(7.5)
time.sleep(2)
print("Servos initialized.")

# Path to the trained YOLO model
path_to_model = "/home/user/Desktop/dart/best.pt"
model = YOLO(path_to_model, 'v8n')  # Load YOLOv8 model

# Initialize video capture
vid = cv2.VideoCapture("/dev/video0")
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Video started")

# PID Control constants
Kp, Ki, Kd = 100, 0.1, 0.5
cx, cy = -1, 1  # Flip signs for servo direction correction

# Low-pass filter coefficient for smoothing
alpha = 0.1  # Change this value to control smoothness; smaller is smoother

# Initialize filtered angles and PID terms
filtered_angle1 = 90  # Neutral position in degrees
filtered_angle2 = 90
integral_x, integral_y = 0, 0
previous_error_x, previous_error_y = 0, 0

def getAdjustment(windowMax, x):
    normalized_adjustment = x / windowMax - 0.5
    adjustment_direction = -1 if normalized_adjustment > 0 else 1
    return abs(round(normalized_adjustment, 1)), adjustment_direction

def set_servo_angle(pwm, angle):
    duty_cycle = 2.5 + (angle / 18.0)
    pwm.ChangeDutyCycle(duty_cycle)

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()  # Draw the detection results on the frame

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get the center of the bounding box
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)

                window = frame.shape
                xmag, xdir = getAdjustment(window[1], center_x)
                ymag, ydir = getAdjustment(window[0], center_y)

                # PID terms for x and y
                error_x = xdir * xmag
                error_y = ydir * ymag

                # Integral accumulation
                integral_x += error_x
                integral_y += error_y

                # Derivative terms
                derivative_x = error_x - previous_error_x
                derivative_y = error_y - previous_error_y

                adj_x = cx * (Kp * error_x + Ki * integral_x + Kd * derivative_x)
                adj_y = cy * (Kp * error_y + Ki * integral_y + Kd * derivative_y)

                previous_error_x = error_x
                previous_error_y = error_y

                target_angle1 = max(0, min(180, filtered_angle1 + adj_x))
                target_angle2 = max(0, min(180, filtered_angle2 + adj_y))
                filtered_angle1 = (alpha * target_angle1) + ((1 - alpha) * filtered_angle1)
                filtered_angle2 = (alpha * target_angle2) + ((1 - alpha) * filtered_angle2)

                # Update servos with filtered angles
                set_servo_angle(horizontal_servo, filtered_angle1)
                set_servo_angle(vertical_servo, filtered_angle2)

        cv2.imshow('Jetson Vision', annotated_frame)  # Show the annotated frame with object detections

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if the user presses 'q'
            break

finally:
    # Release resources
    vid.release()
    horizontal_servo.stop()
    vertical_servo.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
