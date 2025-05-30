# lane_node.py

## Overview
`lane_node.py` is a ROS2 node that performs lane detection using a YOLOPv2 model. It subscribes to camera images from the `/carla/camera` topic, runs lane detection, and publishes the detected lane lines as a JSON string to the `/perception/lane_alone` topic.

---

## Subscribed Topic
- **`/carla/camera`** (`sensor_msgs/msg/Image`):
  - The node receives RGB images from this topic for lane detection.

## Published Topic
- **`/perception/lane_alone`** (`std_msgs/msg/String`):
  - The node publishes detected lane lines as a JSON-encoded string.

---

## Output Format
The published message on `/perception/lane_alone` is a `std_msgs/String` containing a JSON object:

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

---

## How to Run
1. **Install dependencies:**
   - ROS2 Foxy
   - Python packages: `torch`, `opencv-python`, `numpy`
   - YOLOPv2 model weights at `data/weights/yolopv2.pt`

2. **Start ROS2 and the Carla bridge (if needed).**

3. **Run the node:**
   ```bash
   python3 lane_node.py
   ```

4. **View the output:**
   ```bash
   ros2 topic echo /perception/lane_alone
   ```

---

## Example Output
```
std_msgs/msg/String:
  data: '{"lane_lines": [[[100, 200], [110, 210], ...], [[300, 400], [310, 410], ...]]}'
```

---

## Requirements
- ROS2 Foxy
- Python 3.8+
- PyTorch (torch)
- OpenCV (opencv-python)
- numpy
- YOLOPv2 model weights at `data/weights/yolopv2.pt`

---

## Notes
- The node expects images of size 640x640 or will resize them automatically.
- The output is a JSON string for easy parsing and integration.
- For best results, ensure the camera is facing a road with visible lane markings.
