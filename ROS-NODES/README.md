# image_processing_node.py

## Overview
`image_processing_node.py` is a ROS2 node that performs both lane detection (using YOLOPv2) and traffic sign detection (using Ultralytics YOLO). It subscribes to camera images from the `/carla/camera` topic, runs both detection tasks, and publishes the results to separate topics as JSON strings.

---

## IMPORTANT NOTE ABOUT TRAFFIC SIGN DETECTION WEIGHTS
Since Derda hasn't provided the yolov11 weights, you can use a pretrained traffic sign detection weight file for yolov4 provided in the link below:
https://github.com/fredotran/traffic-sign-detector-yolov4/releases/download/weights/yolov4-rds_best_2000.weights

NOT: .weights dosyası olduğu içi çalışmıyor. .pt dosyasına dönüştürülmeli.

## Subscribed Topic
- **`/carla/camera`** (`sensor_msgs/msg/Image`):
  - The node receives RGB images from this topic for processing.

## Published Topics
- **`/perception/lane`** (`std_msgs/msg/String`):
  - Publishes detected lane lines as a JSON-encoded string.
- **`/perception/traffic_signs`** (`std_msgs/msg/String`):
  - Publishes detected traffic signs as a JSON-encoded string.

---

## Output Format
### Lane Detection (`/perception/lane`)
The published message is a `std_msgs/String` containing a JSON object:
```json
{
  "lane_lines": [
    [[x1, y1], [x2, y2], ..., [xn, yn]],
    ...
  ]
}
```
- Each element in `lane_lines` is a list of `[x, y]` points representing a detected lane line (as a polyline).
- If no lanes are detected, `lane_lines` will be an empty list: `{ "lane_lines": [] }`

### Traffic Sign Detection (`/perception/traffic_signs`)
The published message is a `std_msgs/String` containing a JSON object:
```json
{
  "traffic_signs": [
    {
      "class": <int>,
      "confidence": <float>,
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ]
}
```
- Each element in `traffic_signs` is a dictionary with the detected class, confidence, and bounding box coordinates.
- If no signs are detected, `traffic_signs` will be an empty list: `{ "traffic_signs": [] }`

---

## How to Run
1. **Install dependencies:**
   - ROS2 Foxy
   - Python packages: `torch`, `opencv-python`, `numpy`, `ultralytics`
   - YOLOPv2 lane detection weights at `data/weights/yolopv2.pt`
   - Ultralytics YOLO traffic sign detection weights at `data/weights/yolov4.weights` (or your chosen weights)

2. **Start ROS2 and the Carla bridge (if needed).**

3. **Run the node:**
   ```bash
   python3 image_processing_node.py
   ```

4. **View the output:**
   ```bash
   ros2 topic echo /perception/lane
   ros2 topic echo /perception/traffic_signs
   ```

---

## Example Output
### Lane Detection
```
std_msgs/msg/String:
  data: '{"lane_lines": [[[100, 200], [110, 210], ...], [[300, 400], [310, 410], ...]]}'
```

### Traffic Sign Detection
```
std_msgs/msg/String:
  data: '{"traffic_signs": [{"class": 1, "confidence": 0.98, "bbox": [100.0, 200.0, 150.0, 250.0]}]}'
```

---

## Requirements
- ROS2 Foxy
- Python 3.8+
- PyTorch (torch)
- OpenCV (opencv-python)
- numpy
- ultralytics
- YOLOPv2 lane detection weights at `data/weights/yolopv2.pt`
- Ultralytics YOLO traffic sign detection weights at `data/weights/yolov4.weights`

---

## Notes
- The node expects images of size 640x640 or will resize them automatically for lane detection.
- The output is a JSON string for easy parsing and integration.
- For best results, ensure the camera is facing a road with visible lane markings and traffic signs. 
