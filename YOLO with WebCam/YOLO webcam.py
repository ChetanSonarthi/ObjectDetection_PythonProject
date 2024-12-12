# Import necessary libraries
from ultralytics import YOLO  # Import YOLO model from ultralytics
import cv2  # OpenCV for image processing
import cvzone  # Cvzone for better visualization, like drawing bounding boxes
import math  # For mathematical operations
import time  # For calculating frame time (FPS)

# Initialize video capture from webcam or video file
# cap = cv2.VideoCapture(0)
# cap.set(3, 1920)  # Set width
# cap.set(4, 1080)  # Set height
cap = cv2.VideoCapture(
    "C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/YOLO with WebCam/Demo-Videos/cars.mp4")  # Use a video file instead of webcam

# Load the YOLOv8 model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Object class names list
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Set Frames Per Second (FPS) calculation variables
prev_frame_time = 0
new_frame_time = 0

# Loop to continuously capture frames from the video
while True:
    new_frame_time = time.time()  # Get the current time
    success, img = cap.read()  # Read a frame from the video
    if not success:  # Stop if the video ends
        break

    results = model(img, stream=True)  # YOLO model to detect objects in the frame

    # Loop through each detected object
    for r in results:
        boxes = r.boxes  # Get the detected boxes (bounding boxes around objects)

        # Loop through each bounding box
        for box in boxes:
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integer values

            # Draw a rounded corner rectangle for the detected object
            w, h = x2 - x1, y2 - y1  # Calculate width & height of the bounding box
            cvzone.cornerRect(img, (x1, y1, w, h))  # Draw bounding box

            # Get confidence score and object class
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimal places
            cls = int(box.cls[0])

            # Display class name and confidence score on the image
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Calculate and display FPS
    fps = int(1 / (new_frame_time - prev_frame_time))  # Calculate FPS
    prev_frame_time = new_frame_time  # Update frame time
    cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Display FPS on image

    # Show the image with detected objects and FPS
    cv2.imshow("Chetan Project", img)
    if cv2.waitKey(0) == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

car_count = 0
people_count = 0

import time

start_time = time.time()

# Simulate detection on a test video
cap = cv2.VideoCapture('test_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # Add validation logic here, such as checking detection accuracy
    results.show()

cap.release()

end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")
