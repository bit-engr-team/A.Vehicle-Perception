#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import json
import torch

# Import lane_line_mask and get_lane_points from LaneDetection.py or utils
# For this example, we assume they are available in the same file or can be copied here

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

# Dummy lane_line_mask for demonstration (replace with actual function)
def lane_line_mask(ll):
    # Assume ll is a torch tensor with shape (1, H, W)
    # Return a numpy array mask (H, W)
    if isinstance(ll, torch.Tensor):
        ll = ll.squeeze().cpu().numpy()
    return (ll > 0.5).astype(np.uint8)

class LaneNode(Node):
    def __init__(self):
        super().__init__('lane_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/camera',
            self.listener_callback,
            10)
        self.lane_pub = self.create_publisher(String, '/perception/lane_alone', 10)
        # Load YOLOPv2 model
        self.model = torch.jit.load('data/weights/yolopv2.pt', map_location='cpu')
        self.model.eval()
        self.get_logger().info('LaneNode initialized with YOLOPv2.')

    def listener_callback(self, msg):
        try:
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return
        # Preprocess for model
        img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        # Model inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img)
        # Lane mask extraction
        ll_seg_mask = lane_line_mask(ll)
        # Lane points extraction
        lane_lines = get_lane_points(ll_seg_mask)
        # Publish
        lane_msg = String()
        lane_msg.data = json.dumps({'lane_lines': lane_lines})
        self.lane_pub.publish(lane_msg)
        self.get_logger().info(f'Published {len(lane_lines)} lanes.')

def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 