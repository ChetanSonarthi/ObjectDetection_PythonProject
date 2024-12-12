from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Path to the image
img_path = "C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/RunnninG YOLO/Images/5.jpg"

# Run the model on the image
results = model(img_path)
annotated_image = results[0].plot()

# Set up the window
window_name = "YOLO Detection Output"
max_width, max_height = 800, 600
image_height, image_width = annotated_image.shape[:2]

# Resize the image if it exceeds the maximum dimensions
if image_width > max_width or image_height > max_height:
    scale = min(max_width / image_width, max_height / image_height)
    annotated_image = cv2.resize(annotated_image, (int(image_width * scale), int(image_height * scale)))

# Create a named window with normal size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow(window_name, annotated_image)

# Resize the window to the maximum dimensions
cv2.resizeWindow(window_name, max_width, max_height)

# Wait for a key press
cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()





# from ultralytics import YOLO
# import cv2
#
# model = YOLO('../Yolo-Weights/yolov8l.pt')
# results = model("C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/RunnninG YOLO/Local Images/2.jpg", show=True)
# cv2.waitKey(0)


# from ultralytics import YOLO
# import cv2
#
# model = YOLO('../Yolo-Weights/yolov8l.pt')
# img_path = "C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/RunnninG YOLO/Local Images/52 (2).jpg"
# results = model(img_path)
# annotated_image = results[0].plot()
#
# window_name = "YOLO Detection Output"
# max_width, max_height = 800, 600
# image_height, image_width = annotated_image.shape[:2]
#
# if image_width > max_width or image_height > max_height:
#     scale = min(max_width / image_width, max_height / image_height)
#     annotated_image = cv2.resize(annotated_image, (int(image_width * scale), int(image_height * scale)))
#
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.imshow(window_name, annotated_image)
# cv2.resizeWindow(window_name, max_width, max_height)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()























# from ultralytics import YOLO
# import cv2
#
# model = YOLO('../Yolo-Weights/yolov8l.pt')
# img_path = "C:/Users/ASUS/Desktop/ObjectDetection_PythonProject/RunnninG YOLO/Local Images/52 (6).jpg"
# results = model(img_path)
# annotated_image = results[0].plot()
#
# window_name = "YOLO Detection Output"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.imshow(window_name, annotated_image)
#
# original_height, original_width = annotated_image.shape[:2]
# cv2.resizeWindow(window_name, original_width, original_height)
#
# while True:
#     key = cv2.waitKey(1)
#     if key == ord('f'):
#         cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     elif key == ord('q'):
#         break

# cv2.destroyAllWindows()
