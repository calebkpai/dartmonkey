import cv2
import cv2.aruco as aruco
from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
import subprocess

# Initialize video capture with lower resolution for faster processing
vid = cv2.VideoCapture('/dev/video0')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Video started")

# ArUco dictionary for 4x4 marker detection
arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
arucoParam = aruco.DetectorParameters_create()

result = subprocess.run(['sudo', 'pigpiod'], check=True, text=True, capture_output=True)
# Initialize servos with PiGPIO factory
pigpio_factory = PiGPIOFactory()
vertical_servo = AngularServo(18, pin_factory=pigpio_factory)
horizontal_servo = AngularServo(13, pin_factory=pigpio_factory)
vertical_servo.angle = 0;
horizontal_servo.angle = 0
sleep(2)
print("Servos initialized.")

# PID Control constants
Kp, Ki, Kd = 35, 0.1, 0.5
cx, cy = -1, 1  # Flip signs for servo direction correction

# Low-pass filter coefficient for smoothing
alpha = 0.7  # Change this value to control smoothness; smaller is smoother

# Initialize filtered angles and PID terms
filtered_angle1 = vertical_servo.angle
filtered_angle2 = horizontal_servo.angle
integral_x, integral_y = 0, 0
previous_error_x, previous_error_y = 0, 0

def findArucoMarkers(img):
    bbox, ids, _ = aruco.detectMarkers(img, arucoDict, parameters=arucoParam)
    if ids is not None:
        aruco.drawDetectedMarkers(img, bbox)
    return bbox, ids

def getAdjustment(windowMax, x):
    normalized_adjustment = x / windowMax - 0.5
    adjustment_direction = -1 if normalized_adjustment > 0 else 1
    return abs(round(normalized_adjustment, 1)), adjustment_direction

# Main loop
while True:
    ret, img = vid.read()
    if not ret:
        break

    # Convert image to grayscale for faster processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bbox, ids = findArucoMarkers(gray_img)
    if ids is not None:
        top_left = bbox[0][0][0][0], bbox[0][0][0][1]
        bottom_right = bbox[0][0][2][0], bbox[0][0][2][1]
        centre = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

        window = gray_img.shape
        xmag, xdir = getAdjustment(window[0], centre[1])
        ymag, ydir = getAdjustment(window[1], centre[0])

        # PID terms for x and y
        error_x = xdir * xmag
        error_y = ydir * ymag

        # Integral accumulation
        integral_x += error_x
        integral_y += error_y

        # Derivative terms
        derivative_x = error_x - previous_error_x
        derivative_y = error_y - previous_error_y

        # PID-based adjustments
        adj_x = cx * (Kp * error_x + Ki * integral_x + Kd * derivative_x)
        adj_y = cy * (Kp * error_y + Ki * integral_y + Kd * derivative_y)

        # Update previous errors
        previous_error_x = error_x
        previous_error_y = error_y

        # Apply the low-pass filter to servo adjustments
        target_angle1 = max(-90, min(90, vertical_servo.angle + adj_x))
        target_angle2 = max(-90, min(90, horizontal_servo.angle + adj_y))
        filtered_angle1 = (alpha * target_angle1) + ((1 - alpha) * filtered_angle1)
        filtered_angle2 = (alpha * target_angle2) + ((1 - alpha) * filtered_angle2)

        # Update servos with filtered angles
        vertical_servo.angle = filtered_angle1
        horizontal_servo.angle = filtered_angle2
        print(f"horiz:{horizontal_servo.angle}\t vert:{vertical_servo.angle}")


        # Display the grayscale camera feed with markers
        cv2.imshow("Grayscale Camera Feed", gray_img)

    # Quit program on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()