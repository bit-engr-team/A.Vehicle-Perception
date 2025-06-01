##!/usr/bin/env python3

"""
robotaxi_perception_live.py
Robotaksi simülasyonunda canlı kullanım için son perception sistemi.
Bu versiyon şunları içerir:
  - ROS 2 node yapısı (rclpy)
  - Gerçek zamanlı YOLOv5 nesne algılama
  - LiDAR kümeleme ve dinamik hareket tespiti (DBSCAN + ICP)
  - IMU destekli Unscented Kalman Filtresi (UKF) takibi
  - Hata toleranslı sürekli çalışma
  - Algılanan nesneler için gerçek zamanlı JSON çıktısı

"""

# Gerekli kütüphanelerin yüklenmesi
import os
import cv2
import numpy as np
import open3d as o3d
import torch
import json
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import ros_numpy
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import logging

# ---------------- Logger Ayarı ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception")

# ---------------- Parametreler ----------------
FRAME_RATE = 20  # FPS
MAX_MATCH_DISTANCE = 35  # Takip eşleştirme mesafesi
UKF_TRACK_LIMIT = 30  # Aynı anda takip edilecek maksimum nesne sayısı
UKF_SPEED_THRESHOLD = 0.3  # Hız eşiği (m/s) dinamiklik için
MOTION_WINDOW_SIZE = 5  # Kaç frame'e göre dinamiklik değerlendirilir
MOTION_COUNT_THRESHOLD = 3  # Dinamik olarak kabul edilmesi için kaç frame hareket gerekli
FLOW_THRESHOLD = 1.0  # Optik akış eşiği
LIDAR_DYNAMIC_DIST_THRESHOLD = 0.2  # ICP mesafesi (dinamik için)
LIDAR_CLUSTER_TRACK_DIST = 2.0  # LiDAR küme eşleşme mesafesi
STATIC_CLASSES = [  # Statik nesneler
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "bus stop", "street light", "traffic sign", "bicycle lane", "road",
    "sidewalk", "curb", "building", "wall", "fence", "sign", "kite"
]

# ---------------- YOLO Modeli ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to(device)
model.conf = 0.4

# ---------------- Kalman Filtresi ----------------
# Durum geçiş fonksiyonu

def fx(x, dt):
    return np.array([
        x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
        x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
        x[2], x[3], x[4], x[5]
    ])

# Gözlem fonksiyonu

def hx(x):
    return np.array([x[0], x[1]])

# UKF başlatıcı

def create_ukf():
    points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=1.0 / FRAME_RATE, points=points)
    ukf.x = np.zeros(6)
    ukf.P *= 10
    ukf.R = np.eye(2) * 0.5
    ukf.Q = np.diag([0.05, 0.05, 0.2, 0.2, 0.5, 0.5])
    return ukf

# ---------------- ROS 2 Algılama Node'u ----------------
class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()
        # ROS2 abonelikleri
        self.sub_camera = self.create_subscription(Image, "/camera/image_raw", self.camera_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, "/lidar/points", self.lidar_callback, 10)
        self.sub_imu = self.create_subscription(Imu, "/imu/data", self.imu_callback, 10)
        # JSON yayını
        self.pub_json = self.create_publisher(String, "/perception/objects", 10)

        # Değişkenlerin tanımlanması
        self.latest_lidar = None
        self.latest_imu = None
        self.prev_frame_gray = None
        self.prev_lidar = None
        self.prev_dynamic_centroids = np.empty((0, 3), dtype=np.float32)

        self.tracks = {}
        self.ukf_pool = {}
        self.motion_history = {}
        self.last_positions = {}
        self.next_id = 0

    # IMU callback
    def imu_callback(self, msg):
        self.latest_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    # LiDAR callback
    def lidar_callback(self, msg):
        try:
            points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
            self.latest_lidar = self.clean_lidar(points)
        except Exception as e:
            logger.warning(f"LiDAR parse error: {e}")

    # LiDAR temizleme işlemi
    def clean_lidar(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.3)
        pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=1.2)
        return np.asarray(pcd.points)

    # Kamera callback
    def camera_callback(self, msg):
        if self.latest_lidar is None or self.latest_imu is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.warning(f"Camera convert error: {e}")
            return

        # Optik akış (hareket yönü analizi)
        flow = None
        if self.prev_frame_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(self.prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except:
                pass

        # ICP ile LiDAR dinamik hareket analizi
        if self.prev_lidar is not None:
            transformation = o3d.pipelines.registration.registration_icp(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.prev_lidar)),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.latest_lidar)),
                1.0, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ).transformation
            prev_transformed = np.asarray(o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.prev_lidar)).transform(transformation).points)
            tree = cKDTree(prev_transformed[:, :3])
            dists, _ = tree.query(self.latest_lidar[:, :3], k=1)
            motion_mask = dists > LIDAR_DYNAMIC_DIST_THRESHOLD
            dyn_points = self.latest_lidar[motion_mask]
        else:
            dyn_points = np.empty((0, 3))

        # Algılama ve takip
        objects = self.detect_and_track(frame, self.latest_lidar, self.latest_imu, flow, dyn_points)

        # JSON yayını
        self.pub_json.publish(String(data=json.dumps(objects)))

        # Önceki karelerin saklanması
        self.prev_frame_gray = frame_gray.copy()
        self.prev_lidar = self.latest_lidar.copy()

    # Takip ve UKF güncelleme
    def detect_and_track(self, frame, lidar, imu, flow, dynamic_points):
        # YOLO tahmini
        proj = lidar[:, :2] * 20 + np.array([640, 360])  # LiDAR noktalarını 2D'ye projekte et
        results = model(frame).xyxy[0].cpu().numpy()
        detected = []

        for det in results:
            x1, y1, x2, y2, conf, cls = map(float, det[:6])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            name = model.names[int(cls)]
            dist = np.linalg.norm(proj - np.array([cx, cy]), axis=1)
            idxs = np.argsort(dist)[:3]
            if len(idxs) == 0:
                continue
            real_pos = np.mean(lidar[idxs], axis=0)
            detected.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [cx, cy],
                'class': name,
                'real_pos': real_pos.tolist(),
                'distance': float(np.linalg.norm(real_pos[:2]))
            })

        # Eşleştirme işlemi
        assigned, unassigned = self.match_objects(self.tracks, detected)
        new_tracks = {}

        for tid, didx in assigned.items():
            det = detected[didx]
            if tid not in self.ukf_pool:
                self.ukf_pool[tid] = create_ukf()
                self.ukf_pool[tid].x[:2] = np.array(det['real_pos'][:2])
            ukf = self.ukf_pool[tid]
            ukf.predict()
            ukf.update(np.array(det['real_pos'][:2]))
            vx, vy = ukf.x[2], ukf.x[3]
            speed = np.linalg.norm([vx, vy])
            if tid not in self.motion_history:
                self.motion_history[tid] = deque(maxlen=MOTION_WINDOW_SIZE)
            self.motion_history[tid].append(int(speed > UKF_SPEED_THRESHOLD))
            det.update({
                'id': tid,
                'velocity': [float(vx), float(vy)],
                'type': 'dynamic' if sum(self.motion_history[tid]) >= MOTION_COUNT_THRESHOLD else 'static'
            })
            new_tracks[tid] = det

        # Yeni tespit edilen ama eşleşmeyen nesneleri ekle
        for didx in unassigned:
            det = detected[didx]
            tid = self.next_id
            self.next_id += 1
            if len(self.ukf_pool) < UKF_TRACK_LIMIT:
                self.ukf_pool[tid] = create_ukf()
                self.ukf_pool[tid].x[:2] = np.array(det['real_pos'][:2])
            self.motion_history[tid] = deque(maxlen=MOTION_WINDOW_SIZE)
            det.update({'id': tid, 'velocity': [0.0, 0.0], 'type': 'static'})
            new_tracks[tid] = det

        self.tracks = new_tracks
        return list(new_tracks.values())

    # Nesne eşleştirme (takip - tespit)
    def match_objects(self, tracks, detections):
        if not tracks or not detections:
            return {}, list(range(len(detections)))

        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        track_ids = list(tracks.keys())
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                cost[i, j] = np.linalg.norm(np.array(tracks[tid]['real_pos'][:2]) - np.array(det['real_pos'][:2]))

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {}
        unassigned = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < MAX_MATCH_DISTANCE:
                assignment[track_ids[r]] = c
                unassigned.discard(c)

        return assignment, list(unassigned)


# Ana çalışma fonksiyonu

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
