import cv2
import cv2.aruco as aruco
import Jetson.GPIO as GPIO
import time

# Initialize video capture with lower resolution for faster processing
vid = cv2.VideoCapture('/dev/video0')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Video started")

# ArUco dictionary for 4x4 marker detection
arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
arucoParam = aruco.DetectorParameters_create()

# Pin configuration
VERTICAL_PIN = 18
HORIZONTAL_PIN = 13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(VERTICAL_PIN, GPIO.OUT)
GPIO.setup(HORIZONTAL_PIN, GPIO.OUT)

# Initialize PWM for servos
vertical_servo = GPIO.PWM(VERTICAL_PIN, 50)  # 50 Hz frequency
horizontal_servo = GPIO.PWM(HORIZONTAL_PIN, 50)

vertical_servo.start(7.5)  # Neutral position (90 degrees)
horizontal_servo.start(7.5)
time.sleep(2)
print("Servos initialized.")

# PID Control constants
Kp, Ki, Kd = 35, 0.1, 0.5
cx, cy = -1, 1  # Flip signs for servo direction correction

# Low-pass filter coefficient for smoothing
alpha = 0.7  # Change this value to control smoothness; smaller is smoother

# Initialize filtered angles and PID terms
filtered_angle1 = 90  # Neutral position in degrees
filtered_angle2 = 90
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

def set_servo_angle(pwm, angle):
    duty_cycle = 2.5 + (angle / 18.0)
    pwm.ChangeDutyCycle(duty_cycle)

# Main loop
try:
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
            target_angle1 = max(0, min(180, filtered_angle1 + adj_x))
            target_angle2 = max(0, min(180, filtered_angle2 + adj_y))
            filtered_angle1 = (alpha * target_angle1) + ((1 - alpha) * filtered_angle1)
            filtered_angle2 = (alpha * target_angle2) + ((1 - alpha) * filtered_angle2)

            # Update servos with filtered angles
            set_servo_angle(vertical_servo, filtered_angle1)
            set_servo_angle(horizontal_servo, filtered_angle2)
            print(f"horiz:{filtered_angle2}\t vert:{filtered_angle1}")

            # Display the grayscale camera feed with markers
            cv2.imshow("Grayscale Camera Feed", gray_img)

        # Quit program on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    vid.release()
    vertical_servo.stop()
    horizontal_servo.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
