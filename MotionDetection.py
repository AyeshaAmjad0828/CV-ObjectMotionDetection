import cv2
import time

# Load video file
video_path = 'PendulumVideos/SwingingPendulum.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Parameters for pendulum detection (you may need to adjust these)
threshold_value = 50
min_contour_area = 100

# Variables for pendulum motion detection
pendulum_motion = False
start_time = None
end_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform thresholding
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter contours by area
        if cv2.contourArea(contour) > min_contour_area:
            pendulum_motion = True
            if start_time is None:
                start_time = time.time()

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Calculate duration of pendulum motion
if pendulum_motion and start_time is not None:
    end_time = time.time()
    duration = end_time - start_time
    print(f"Pendulum motion duration: {duration} seconds")
else:
    print("No pendulum motion detected in the video.")
