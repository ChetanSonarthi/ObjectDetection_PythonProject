# Import necessary libraries

from cvzone import cornerRect
from ultralytics import YOLO  # Import YOLO model from ultralytics
import cv2  # OpenCV for image processing
import cvzone  # Cvzone for better visualization, like drawing bounding boxes
import math  # For mathematical operations
import time  # For calculating frame time (FPS)

from sort import * #FOR OBJECT TRACKING

cap = cv2.VideoCapture("C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/YOLO with WebCam/Demo-Videos/cars.mp4")  # use a video file instead of webcam

# Load the YOLOv8 model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# class names (objects)
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

mask = cv2.imread("MASK.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


limits = [400, 297, 673, 297]
totalCount = []

# loop to continuously capture frames from the source
while True:
    success, img = cap.read()  # Read a frame
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(imgRegion, stream=True)  # YOLO model to detect objects in the frame

    detections = np.empty((0, 5))

    # Loop through each result (detect)
    for r in results:
        boxes = r.boxes  # Get the detected boxes (bounding boxes around objects)

        # Loop bounding box
        for box in boxes:

            # coordinates of the bounding box----------------------(top-left corner and bottom-right corner)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer values

            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)--FOR OPEN CV

            # cvzone.cornerRect draws a rectangle with rounded corners around the detected object

            w, h = x2 - x1, y2 - y1  # Calculate width & height of the BB
            cvzone.cornerRect(img, (x1, y1, w, h),l=7)  # Draw the bounding box

            # Confidence score ----------------------(how confident YOLO is about the detected object)
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimal places

            # Class of the detected object-------(e.g., person, car)
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:

                    #Display class name and confidence score on the image next to the detected object
                    # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=0.7, thickness=1,offset=5)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 189), 4)


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #Float--->Int
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(198, 85, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=3, offset=7)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (200, 68), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5)


    # Resulting image with detected objects and bounding boxes

    cv2.imshow("Chetan Project", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)  # Refresh the image






