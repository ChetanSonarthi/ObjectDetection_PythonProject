# Object Detection for Traffic and Crowd Analysis

## 📌 Project Overview
This project implements **real-time object detection** for **car and people counting** using **YOLO** and **OpenCV** in Python. It is designed for **traffic monitoring** and **crowd analysis**, providing accurate detection and tracking of objects in videos and webcam feeds.

## 🚀 Features
- **Car and People Counter** – Detects and counts cars & pedestrians.
- **YOLO Integration** – Uses pre-trained YOLO weights for high-speed detection.
- **Webcam & Video Support** – Works on live streams and video files.
- **SQLite Integration** – Stores detection logs for future analysis.
- **SORT Tracking** – Implements **SORT (Simple Online and Realtime Tracker)** for object tracking.
- **Real-World Dataset** – Utilizes an **Indian dataset** for better accuracy in local scenarios.

## 🏗️ Project Structure
```
ObjectDetection_PythonProject/
│── Car Counter Project 1/
│── People Counter Project 2/
│── Running YOLO/
│── YOLO with Webcam/
│── Yolo-Weights/
│── main.py  # Main script for object detection
│── requirements.txt  # Dependencies
│── sort.py  # SORT tracking algorithm (GPL v3 Licensed)
│── .gitignore
│── LICENSE (If applicable)
│── README.md (This file)
```

## 🛠️ Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/ChetanSonarthi/ObjectDetection_PythonProject.git
cd ObjectDetection_PythonProject
```
### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run Object Detection**
- **For YOLO on a video file:**
  ```sh
  python main.py --video input_video.mp4
  ```
- **For YOLO with webcam:**
  ```sh
  python main.py --webcam
  ```

## 📂 Dependencies
- **Python 3.8+**
- **OpenCV**
- **YOLOv8**
- **NumPy**
- **SQLite3**
- **SORT Tracker** *(GPL v3 licensed)*

## 🎥 YouTube Video & Website
- **YouTube Demo:** [[YouTube Video Link Here](https://youtu.be/WgPbbWmnXJ8?si=HLHE0jM1RWhEvqLu)]
- **Project Website:** [[Website Link Here](https://www.computervision.zone/courses/object-detection-course/)]

## ⚖️ Licensing & Legal Considerations
- This project includes **sort.py**, which is licensed under **GNU GPL v3**. This means **any modifications** or **redistributions** of this script must also be **GPL-compliant**.
- Ensure proper compliance with all **local and international regulations** before deployment.

## 📢 Acknowledgments
- **SORT Tracker** by [Alex Bewley](https://github.com/abewley/sort) *(GPL v3 License)*
- **YOLO Object Detection** – Inspired by open-source implementations.
- **Dataset** – Custom dataset for localized performance.

## 📜 Terms of Use
- This project is intended for **educational and research purposes only**.
- Users must comply with all **applicable legal and ethical standards** when deploying the software.
- **No liability** is assumed for any misuse of this project.

## 📬 Contact
For questions or contributions, feel free to reach out via [GitHub Issues](https://github.com/ChetanSonarthi/ObjectDetection_PythonProject/issues).



