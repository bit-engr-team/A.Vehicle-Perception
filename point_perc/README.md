# Sensor Fusion Based Object Tracking

ŞU ANDA LIVE MODE YOK, YANİ DİREKT ROS'TAN GELEN VERİYİ DİNLEYEMİYOR, SADECE DOSYA OKUYOR


This project performs **real-time object tracking and classification** by fusing data from **Camera**, **LiDAR**, and **IMU** sensors.  
It detects objects with **YOLOv5**, tracks them using an **Unscented Kalman Filter (UKF)**, and classifies them as **Dynamic** or **Static** based on their motion.

## Features

- 🚗 YOLOv5n (lightweight) object detection
- 🛰️ LiDAR point cloud cleaning and 2D projection
- 🧭 IMU acceleration data integration
- 🔵 Unscented Kalman Filter (UKF) based tracking
- 🔴 Dynamic vs Static object classification
- 📄 Outputs tracking results to **CSV** and **JSON**
- 🎥 Real-time visualization with OpenCV


## How It Works

1. **Load IMU, LiDAR, and Camera data** frame-by-frame.
2. **Clean** LiDAR points (remove noise, limit range).
3. **Detect** objects with YOLOv5 on camera images.
4. **Associate** detected objects with closest LiDAR points.
5. **Fuse** sensor data with an Unscented Kalman Filter (UKF).
6. **Classify** objects as Static or Dynamic based on estimated velocity.
7. **Track and update** object information across frames.
8. **Save** results into CSV and JSON files.
9. **Display** live results with object bounding boxes and info.

## Requirements

- Python 3.8+
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- OpenCV
- NumPy
- pandas
- open3d
- scipy
- filterpy
- torch

You can install requirements with:

```bash
pip install torch torchvision torchaudio
pip install opencv-python pandas open3d scipy filterpy
