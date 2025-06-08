#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu, Image
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import deque
from sensor_msgs_py import point_cloud2
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
import io
import cv2
import logging

# --- Parametreler ---
LIDAR_DYNAMIC_DIST_THRESHOLD = 0.2
CLUSTER_EPS = 0.5
CLUSTER_MIN_POINTS = 5
MAX_MATCH_DISTANCE = 2.0
UKF_SPEED_THRESHOLD = 0.3
MOTION_WINDOW_SIZE = 5
MOTION_COUNT_THRESHOLD = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_detection")

def fx(x, dt):
    # IMU ivmesi eklenirse buraya eklenebilir
    return np.array([
        x[0] + x[2] * dt,
        x[1] + x[3] * dt,
        x[2],
        x[3]
    ])

def hx(x):
    return np.array([x[0], x[1]])

def create_ukf():
    points = MerweScaledSigmaPoints(n=4, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=0.1, points=points)
    ukf.x = np.zeros(4)
    ukf.P *= 10
    ukf.R = np.eye(2) * 0.5
    ukf.Q = np.diag([0.05, 0.05, 0.2, 0.2])
    return ukf

class LidarImuDynamicDetector(Node):
    def __init__(self):
        super().__init__('lidar_imu_dynamic_detector')
        self.sub_lidar = self.create_subscription(PointCloud2, "/carla/lidar", self.lidar_callback, 10)
        self.sub_imu = self.create_subscription(Imu, "/carla/imu", self.imu_callback, 10)
        self.pub = self.create_publisher(Image, "/dynamic_objects_image", 10)
        self.prev_points = None
        self.tracks = {}
        self.motion_history = {}
        self.next_id = 0
        self.latest_imu = np.zeros(2)  # [ax, ay]
        self.lidar_buffer = deque(maxlen=10)  # Son 10 lidar verisi için buffer

    def imu_callback(self, msg):
        # Sadece x ve y ivmesi
        self.latest_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    def clean_lidar(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.3)
        pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=1.2)
        arr = np.asarray(pcd.points)
        return arr[arr[:,2] > 0.2] if arr.shape[0] > 0 else arr

    def lidar_callback(self, msg):
        points = np.array([
            [p[0], p[1], p[2]]
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ], dtype=np.float32)
        points = self.clean_lidar(points)
        if points.shape[0] == 0:
            return

        # Son 10 lidar verisini birleştir
        self.lidar_buffer.append(points)
        if len(self.lidar_buffer) < 10:
            self.prev_points = points
            return
        merged_points = np.vstack(self.lidar_buffer)

        if self.prev_points is None or self.prev_points.shape[0] == 0:
            self.prev_points = merged_points
            return

        # ICP ile hizalama
        pcd_prev = o3d.geometry.PointCloud()
        pcd_prev.points = o3d.utility.Vector3dVector(self.prev_points)
        pcd_curr = o3d.geometry.PointCloud()
        pcd_curr.points = o3d.utility.Vector3dVector(merged_points)
        threshold = 1.0
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_prev, pcd_curr, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        transformation = reg_p2p.transformation
        pcd_prev.transform(transformation)
        aligned_prev = np.asarray(pcd_prev.points)

        # Dinamik noktaları bul
        tree = cKDTree(aligned_prev)
        dists, _ = tree.query(merged_points, k=1)
        dynamic_mask = dists > LIDAR_DYNAMIC_DIST_THRESHOLD
        dynamic_points = merged_points[dynamic_mask]

        # Kümeleme (DBSCAN)
        cluster_centroids = []
        if dynamic_points.shape[0] > 0:
            logger.info(f"LiDAR ile {dynamic_points.shape[0]} adet dinamik nokta tespit edildi.")
            pcd_dyn = o3d.geometry.PointCloud()
            pcd_dyn.points = o3d.utility.Vector3dVector(dynamic_points)
            labels = np.array(pcd_dyn.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=False))
            for lbl in np.unique(labels[labels >= 0]):
                pts = dynamic_points[labels == lbl]
                centroid = np.mean(pts, axis=0)
                cluster_centroids.append(centroid)
        cluster_centroids = np.array(cluster_centroids)

        # Takip ve Dinamiklik Kararı
        assigned, unassigned = self.match_clusters(self.tracks, cluster_centroids)
        new_tracks = {}

        for tid, cidx in assigned.items():
            centroid = cluster_centroids[cidx]
            ukf = self.tracks[tid]['ukf']
            ukf.predict()
            ukf.update(centroid[:2])
            vx, vy = ukf.x[2], ukf.x[3]
            speed = np.linalg.norm([vx, vy])
            if tid not in self.motion_history:
                self.motion_history[tid] = deque(maxlen=MOTION_WINDOW_SIZE)
            self.motion_history[tid].append(int(speed > UKF_SPEED_THRESHOLD))
            is_dynamic = sum(self.motion_history[tid]) >= MOTION_COUNT_THRESHOLD
            new_tracks[tid] = {
                'ukf': ukf,
                'centroid': centroid,
                'velocity': [float(vx), float(vy)],
                'type': 'dynamic' if is_dynamic else 'static'
            }
            if is_dynamic:
                logger.info(f"Dinamik nesne tespit edildi! id={tid}, pos={centroid[:2]}, hız={speed:.2f} m/s")

        for cidx in unassigned:
            centroid = cluster_centroids[cidx]
            tid = self.next_id
            self.next_id += 1
            ukf = create_ukf()
            ukf.x[:2] = centroid[:2]
            self.motion_history[tid] = deque(maxlen=MOTION_WINDOW_SIZE)
            new_tracks[tid] = {
                'ukf': ukf,
                'centroid': centroid,
                'velocity': [0.0, 0.0],
                'type': 'static'
            }

        self.tracks = new_tracks

        # Sadece dinamik nesneleri çiz
        dynamic_positions = []
        for tid, track in self.tracks.items():
            if track['type'] == 'dynamic':
                dynamic_positions.append(track['centroid'][:2])

        # Görselleştir ve Image olarak yayınla
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(0, 0, c='red', marker='x', label='Merkez (0,0)')
        # Dinamik noktaları (ham) çiz
        if dynamic_points.shape[0] > 0:
            ax.scatter(dynamic_points[:, 0], dynamic_points[:, 1], c='blue', s=5, alpha=0.5, label='Dinamik Noktalar')
        # Dinamik küme merkezlerini çiz
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            ax.scatter(dynamic_positions[:, 0], dynamic_positions[:, 1], c='black', label='Dinamik Küme Merkezleri')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Dinamik Nesneler')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        # Sabit ölçek ve merkez için:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks(np.arange(-10, 11, 2))
        ax.set_yticks(np.arange(-10, 11, 2))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # sensor_msgs.msg.Image'e çevir
        img_msg = Image()
        img_msg.header = msg.header
        img_msg.height, img_msg.width, _ = img.shape
        img_msg.encoding = 'bgr8'
        img_msg.data = img.tobytes()
        img_msg.step = img_msg.width * 3

        self.pub.publish(img_msg)
        self.prev_points = merged_points

    def match_clusters(self, tracks, cluster_centroids):
        if not tracks or len(cluster_centroids) == 0:
            return {}, list(range(len(cluster_centroids)))
        from scipy.optimize import linear_sum_assignment
        track_ids = list(tracks.keys())
        cost = np.zeros((len(track_ids), len(cluster_centroids)), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            prev_pos = tracks[tid]['centroid'][:2]
            for j, centroid in enumerate(cluster_centroids):
                cost[i, j] = np.linalg.norm(prev_pos - centroid[:2])
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {}
        unassigned = set(range(len(cluster_centroids)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < MAX_MATCH_DISTANCE:
                assignment[track_ids[r]] = c
                unassigned.discard(c)
        return assignment, list(unassigned)

def main():
    rclpy.init()
    node = LidarImuDynamicDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()