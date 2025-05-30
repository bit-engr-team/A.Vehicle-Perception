#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import json
import torch
from ultralytics import YOLO

# Lane detection helpers (from LaneDetection.py, simplified)
def get_lane_points(ll_seg_mask, min_area=100):
    lane_lines = []
    ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(ll_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        points = cnt.squeeze()
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if len(points) >= 2:
            x = points[:, 0]
            y = points[:, 1]
            try:
                poly_coeffs = np.polyfit(y, x, 2)
                poly_func = np.poly1d(poly_coeffs)
                y_smooth = np.linspace(y.min(), y.max(), num=50)
                x_smooth = poly_func(y_smooth)
                smooth_line = np.column_stack((x_smooth, y_smooth)).astype(int)
                lane_lines.append(smooth_line.tolist())
            except np.linalg.LinAlgError:
                continue
    return lane_lines

# Lane mask extraction helper (copied from lane_node.py)
def lane_line_mask(ll):
    if isinstance(ll, torch.Tensor):
        ll = ll.squeeze().cpu().numpy()
    return (ll > 0.5).astype(np.uint8)

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/camera',
            self.listener_callback,
            10)
        self.lane_pub = self.create_publisher(String, '/perception/lane', 10)
        self.sign_pub = self.create_publisher(String, '/perception/traffic_signs', 10)
        # Load YOLO model for traffic sign detection
        self.yolo_model = YOLO('data/weights/yolov11.pt')  # Derda yolov11.pt weight dosyasını ekleyecek. Şimdilik başka bir weight dosyası kullanılabilir.
        # Load Lane Detection Model (YOLOPv2 TorchScript)
        self.lane_model = torch.jit.load('data/weights/yolopv2.pt', map_location='cpu')
        self.lane_model.eval()
        self.get_logger().info('ImageProcessingNode initialized.')

    def listener_callback(self, msg):
        # Convert ROS Image to OpenCV image
        try:
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # --- Lane Detection (YOLOPv2) ---
        img_lane = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_lane = cv2.resize(img_lane, (640, 640))
        img_lane = img_lane.transpose(2, 0, 1)  # HWC to CHW
        img_lane = np.ascontiguousarray(img_lane)
        img_lane = torch.from_numpy(img_lane).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.lane_model(img_lane)
        ll_seg_mask = lane_line_mask(ll)
        lane_lines = get_lane_points(ll_seg_mask)
        lane_msg = String()
        lane_msg.data = json.dumps({'lane_lines': lane_lines})
        self.lane_pub.publish(lane_msg)

        # --- Traffic Sign Detection (Ultralytics YOLO) ---
        results = self.yolo_model.predict(img_np, conf=0.25, device='cpu')
        sign_list = []
        for r in results:
            for box in r.boxes:
                sign = {
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': [float(x) for x in box.xyxy[0].tolist()]
                }
                sign_list.append(sign)
        sign_msg = String()
        sign_msg.data = json.dumps({'traffic_signs': sign_list})
        self.sign_pub.publish(sign_msg)

        self.get_logger().info(f'Published {len(lane_lines)} lanes, {len(sign_list)} signs.')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
