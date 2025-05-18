#!/usr/bin/env python3

import rospy
import cv2
import json
from datetime import datetime
from std_msgs.msg import String
from ultralytics import YOLO

# YOLOv11 modelini yükle
best_weights = "runs/train/weights/best.pt"
model = YOLO(best_weights)

# ROS Publisher
pub = rospy.Publisher('/traffic_signs', String, queue_size=10)
rospy.init_node('yolo11_traffic_publisher', anonymous=True)
rate = rospy.Rate(10)  # 10 Hz

# Kamera aç
cap = cv2.VideoCapture(0)

while not rospy.is_shutdown() and cap.isOpened():
    
  ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width  = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2
            center_y = y1 + height / 2

            message = {
                "timestamp": datetime.utcnow().isoformat(),
                "class_name": class_name,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "width": round(width, 2),
                "height": round(height, 2)
            }

            ros_msg = String()
            ros_msg.data = json.dumps(message)
            pub.publish(ros_msg)
            rospy.loginfo(ros_msg.data)

    rate.sleep()

cap.release()

