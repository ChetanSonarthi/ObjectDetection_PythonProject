import cv2
import torch

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8', pretrained=True)

# Initialize video capture
cap = cv2.VideoCapture('traffic_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame)

    # Process detection results
    labels, cords = results.names, results.xywh[0]  # Object names and coordinates

    # Draw bounding boxes
    for cord in cords:
        x, y, w, h = cord[0], cord[1], cord[2], cord[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, labels[int(cord[4])], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
#
# # Load your local video
# cap = cv2.VideoCapture("C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/YOLO with WebCam/Demo-Videos/people.mp4")
#
# # Check if the video was successfully opened
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
#
# # Initialize points for line drawing
# points = []
#
# # Callback function to capture mouse click coordinates
# def select_point(event, x, y, flags, param):
#     global points
#     if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click to select points
#         points.append((x, y))
#         if len(points) == 2:  # If two points are selected, draw line
#             cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
#             print(f"Line coordinates: Start {points[0]}, End {points[1]}")
#             cv2.imshow("Frame", frame)
#             points = []  # Reset after drawing
#
# # Set up the window and mouse callback function
# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", select_point)
#
# print("Click on two points to draw a line. Press 'q' to exit.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or error.")
#         break
#
#     # Show selected coordinates (if any)
#     for i, point in enumerate(points):
#         cv2.circle(frame, point, 5, (0, 0, 255), -1)
#         cv2.putText(frame, f"{point}", (point[0] + 10, point[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#
#     cv2.imshow("Frame", frame)
#
#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/YOLO with WebCam/Demo-Videos/people.mp4











