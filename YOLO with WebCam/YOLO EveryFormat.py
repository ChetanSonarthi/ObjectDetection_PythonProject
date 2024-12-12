from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture for the video file
cap = cv2.VideoCapture("C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/YOLO with WebCam/Local-Videos/ISTC10.mp4")

# Check if video capture was initialized successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load YOLOv8 model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names
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

# Define the display window name
window_name = "Chetan Every Format Detection"

# Set up initial frame time for FPS calculation
prev_frame_time = 0

# Set up the window for normal or fullscreen display
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Main loop
while True:
    # Capture each frame
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Process the frame with YOLOv8 model
    results = model(img, stream=True)

    # Loop through each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"

            # Draw bounding box and label
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2)  # Thicker line for better quality
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=2)  # Larger text and thicker for quality

    # Get original frame dimensions
    frame_height, frame_width = img.shape[:2]

    # Get the screen resolution
    screen_width = 1920  # Set to your screen width
    screen_height = 1080  # Set to your screen height

    # Calculate the scaling factor and aspect ratio
    aspect_ratio = frame_width / frame_height
    if frame_width > screen_width or frame_height > screen_height:
        # Scale down if needed while maintaining aspect ratio
        if frame_width / screen_width > frame_height / screen_height:
            display_width = screen_width
            display_height = int(screen_width / aspect_ratio)
        else:
            display_height = screen_height
            display_width = int(screen_height * aspect_ratio)
    else:
        display_width, display_height = frame_width, frame_height

    # Resize only for display without affecting quality
    display_img = cv2.resize(img, (display_width, display_height))

    # Create black background and center the resized image in it
    centered_display = cv2.copyMakeBorder(display_img,
                                          (screen_height - display_height) // 2,
                                          (screen_height - display_height) // 2,
                                          (screen_width - display_width) // 2,
                                          (screen_width - display_width) // 2,
                                          cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))

    # Show in a window that allows for fullscreen toggle
    cv2.imshow(window_name, centered_display)

    # Set 'f' key to toggle fullscreen and 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('f'):
        # Toggle between normal and fullscreen
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
