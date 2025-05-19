#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO

class Yolo11Ros2Detector(Node):
    def __init__(self):
        super().__init__('yolo11_ros2_detector')
        # Parametreler
        self.declare_parameter('weights_path', '/path/to/your/best.pt')
        self.declare_parameter('image_topic', '/zed2/left/image_rect_color')
        self.declare_parameter('detection_topic', '/traffic_sign_detections')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('inference_rate', 10.0)  # Hz

        weights = self.get_parameter('weights_path').get_parameter_value().string_value
        img_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        det_topic = self.get_parameter('detection_topic').get_parameter_value().string_value

        # Modeli yükle
        self.model = YOLO(weights)
        self.model.conf = float(self.get_parameter('confidence_threshold').get_parameter_value().double_value)
        self.model.iou = 0.45  # istenirse iou da parametre yapılabilir

        # Bridge ve Publisher/Subscriber
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, img_topic, self.image_callback,
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self.pub = self.create_publisher(
            Detection2DArray, det_topic,
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )

        # Zamanlayıcı (opsiyonel, eğer throttle gerekiyorsa)
        period = 1.0 / self.get_parameter('inference_rate').get_parameter_value().double_value
        self.timer = self.create_timer(period, lambda: None)

        self.get_logger().info(f"Yolo11Ros2Detector başlatıldı. Sub:{img_topic} Pub:{det_topic}")

    def image_callback(self, msg: Image):
        # CvBridge ile ROS Image → OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO inference
        results = self.model(cv_img)[0]  # batch size=1

        det_array = Detection2DArray()
        det_array.header = msg.header  # timestamp + frame_id

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # Bounding box merkez ve boyut
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1)
            h  = (y2 - y1)

            det = Detection2D()
            det.header = msg.header

            det.bbox = BoundingBox2D()
            det.bbox.center.x = float(cx)
            det.bbox.center.y = float(cy)
            det.bbox.size_x   = float(w)
            det.bbox.size_y   = float(h)

            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = cls_name
            hypo.hypothesis.score    = conf
            det.results = [hypo]

            det_array.detections.append(det)

        # Yayımla
        self.pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = Yolo11Ros2Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
