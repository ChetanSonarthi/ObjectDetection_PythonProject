from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1920)
cap.set(4, 1080)

# Load the YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names for object detection
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

prev_frame_time = 0
new_frame_time = 0

# Variable to track fullscreen mode
fullscreen = False

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        break

    # Perform object detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))

    # Set the window to fullscreen if enabled
    if fullscreen:
        cv2.namedWindow('Camera Feed', cv2.WND_PROP_FULLSCREEN)  # Set window to fullscreen
        cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Fullscreen mode
    else:
        cv2.namedWindow('Camera Feed')  # Regular window
        cv2.setWindowProperty('Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # Normal window mode

    # Show the output image
    cv2.imshow("Camera Feed", img)

    # Key controls for toggling fullscreen
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('f'):  # Press 'f' to toggle fullscreen
        fullscreen = not fullscreen  # Toggle fullscreen state

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
