#!/usr/bin/env python3
"""
cursor ile optimize edildi
kalman , ror , outlier ror , ıcp parametreleri iyileştirildi 
"""

from __future__ import annotations
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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Deque
import time

# Constants
class DetectionConfig:
    """Configuration parameters for object detection."""
    LIDAR_DYNAMIC_DIST_THRESHOLD: float = 0.2
    CLUSTER_EPS: float = 0.5
    CLUSTER_MIN_POINTS: int = 5
    MAX_MATCH_DISTANCE: float = 2.0
    UKF_SPEED_THRESHOLD: float = 0.3
    MOTION_WINDOW_SIZE: int = 5
    MOTION_COUNT_THRESHOLD: int = 3
    VOXEL_SIZE: float = 0.3
    OUTLIER_RADIUS: float = 1.2
    OUTLIER_MIN_POINTS: int = 8
    MIN_HEIGHT_THRESHOLD: float = 0.2
    LIDAR_BUFFER_SIZE: int = 10
    ICP_THRESHOLD: float = 1.0
    # New ROR parameters
    ROR_ADAPTIVE_RADIUS: bool = True
    ROR_MIN_RADIUS: float = 0.5
    ROR_MAX_RADIUS: float = 2.0
    ROR_STD_MULTIPLIER: float = 1.5
    ROR_PARALLEL_THRESHOLD: int = 10000
    ROR_STATISTICAL_K: int = 20
    ROR_STATISTICAL_STD: float = 2.0

@dataclass
class TrackedObject:
    """Data class for tracked objects."""
    ukf: UnscentedKalmanFilter
    centroid: np.ndarray
    velocity: Tuple[float, float]
    type: str
    last_update: float

class SensorFusion:
    """Handles sensor fusion operations."""
    
    @staticmethod
    def create_ukf() -> UnscentedKalmanFilter:
        """Creates and initializes an Unscented Kalman Filter."""
        points = MerweScaledSigmaPoints(n=4, alpha=1e-3, beta=2., kappa=0)
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=SensorFusion.fx, hx=SensorFusion.hx, dt=0.1, points=points)
        ukf.x = np.zeros(4)
        ukf.P *= 10
        ukf.R = np.eye(2) * 0.5
        ukf.Q = np.diag([0.05, 0.05, 0.2, 0.2])
        return ukf

    @staticmethod
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        """State transition function for UKF."""
        return np.array([
            x[0] + x[2] * dt,
            x[1] + x[3] * dt,
            x[2],
            x[3]
        ])

    @staticmethod
    def hx(x: np.ndarray) -> np.ndarray:
        """Measurement function for UKF."""
        return x[:2]

class PointCloudProcessor:
    """Handles LiDAR point cloud processing."""
    
    @staticmethod
    def _compute_adaptive_radius(points: np.ndarray) -> float:
        """Computes adaptive radius based on point cloud density."""
        if points.shape[0] < 100:  # Too few points for reliable density estimation
            return DetectionConfig.OUTLIER_RADIUS
            
        # Compute average nearest neighbor distance
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2 because first point is the point itself
        avg_dist = np.mean(distances[:, 1])
        
        # Compute standard deviation of distances
        std_dist = np.std(distances[:, 1])
        
        # Adaptive radius based on local density
        adaptive_radius = avg_dist * DetectionConfig.ROR_STD_MULTIPLIER + std_dist
        
        # Clamp to min/max values
        return np.clip(
            adaptive_radius,
            DetectionConfig.ROR_MIN_RADIUS,
            DetectionConfig.ROR_MAX_RADIUS
        )

    @staticmethod
    def _remove_statistical_outliers(points: np.ndarray) -> np.ndarray:
        """Removes statistical outliers before ROR for better performance."""
        if points.shape[0] < DetectionConfig.ROR_STATISTICAL_K:
            return points
            
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=DetectionConfig.ROR_STATISTICAL_K)
        mean_dist = np.mean(distances, axis=1)
        std_dist = np.std(distances, axis=1)
        
        # Points that are within std_multiplier standard deviations
        mask = mean_dist < np.mean(mean_dist) + DetectionConfig.ROR_STATISTICAL_STD * np.std(mean_dist)
        return points[mask]

    @staticmethod
    def _parallel_ror(points: np.ndarray, radius: float, min_points: int) -> np.ndarray:
        """Parallel implementation of Radius Outlier Removal."""
        if points.shape[0] < DetectionConfig.ROR_PARALLEL_THRESHOLD:
            # Use Open3D's implementation for small point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
            return np.asarray(pcd.points)
            
        # For large point clouds, use parallel processing
        import multiprocessing as mp
        from functools import partial
        
        def process_chunk(chunk_points: np.ndarray, tree: cKDTree, radius: float, min_points: int) -> np.ndarray:
            indices = tree.query_ball_point(chunk_points, radius)
            return chunk_points[[len(idx) >= min_points for idx in indices]]
        
        # Split points into chunks for parallel processing
        n_cores = mp.cpu_count()
        chunk_size = points.shape[0] // n_cores
        chunks = [points[i:i + chunk_size] for i in range(0, points.shape[0], chunk_size)]
        
        # Create shared KD-tree
        tree = cKDTree(points)
        
        # Process chunks in parallel
        with mp.Pool(n_cores) as pool:
            process_func = partial(process_chunk, tree=tree, radius=radius, min_points=min_points)
            results = pool.map(process_func, chunks)
        
        # Combine results
        return np.vstack(results) if results else np.array([])

    @staticmethod
    def clean_point_cloud(points: np.ndarray) -> np.ndarray:
        """Cleans and filters point cloud data with optimized ROR."""
        if points.shape[0] == 0:
            return points
            
        # Step 1: Voxel downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(DetectionConfig.VOXEL_SIZE)
        points = np.asarray(pcd.points)
        
        if points.shape[0] == 0:
            return points
            
        # Step 2: Statistical outlier removal (pre-filter)
        points = PointCloudProcessor._remove_statistical_outliers(points)
        
        if points.shape[0] == 0:
            return points
            
        # Step 3: Compute adaptive radius if enabled
        radius = (PointCloudProcessor._compute_adaptive_radius(points) 
                 if DetectionConfig.ROR_ADAPTIVE_RADIUS 
                 else DetectionConfig.OUTLIER_RADIUS)
        
        # Step 4: Apply ROR with parallel processing for large point clouds
        points = PointCloudProcessor._parallel_ror(
            points,
            radius,
            DetectionConfig.OUTLIER_MIN_POINTS
        )
        
        # Step 5: Height threshold filtering
        return points[points[:, 2] > DetectionConfig.MIN_HEIGHT_THRESHOLD] if points.shape[0] > 0 else points

    @staticmethod
    def align_point_clouds(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aligns two point clouds using ICP."""
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(source)
        
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target)
        
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            DetectionConfig.ICP_THRESHOLD,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        pcd_source.transform(reg_p2p.transformation)
        return np.asarray(pcd_source.points), reg_p2p.transformation

class LidarImuDynamicDetector(Node):
    """Main class for dynamic object detection using LiDAR and IMU data."""
    
    def __init__(self):
        super().__init__('lidar_imu_dynamic_detector')
        
        # Initialize subscribers and publishers
        self.sub_lidar = self.create_subscription(
            PointCloud2, "/carla/lidar", self.lidar_callback, 10
        )
        self.sub_imu = self.create_subscription(
            Imu, "/carla/imu", self.imu_callback, 10
        )
        self.pub = self.create_publisher(
            Image, "/dynamic_objects_image", 10
        )
        
        # Initialize state variables
        self.prev_points: Optional[np.ndarray] = None
        self.tracks: Dict[int, TrackedObject] = {}
        self.motion_history: Dict[int, Deque[int]] = {}
        self.next_id: int = 0
        self.latest_imu: np.ndarray = np.zeros(2)
        self.lidar_buffer: Deque[np.ndarray] = deque(maxlen=DetectionConfig.LIDAR_BUFFER_SIZE)
        
        # Setup logging
        self.logger = logging.getLogger("dynamic_detection")
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Dynamic object detector initialized")

    def imu_callback(self, msg: Imu) -> None:
        """Callback for IMU data."""
        self.latest_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    def lidar_callback(self, msg: PointCloud2) -> None:
        """Main callback for LiDAR data processing."""
        try:
            # Process incoming point cloud
            points = np.array([
                [p[0], p[1], p[2]]
                for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            ], dtype=np.float32)
            
            points = PointCloudProcessor.clean_point_cloud(points)
            if points.shape[0] == 0:
                return

            # Update buffer and check if we have enough data
            self.lidar_buffer.append(points)
            if len(self.lidar_buffer) < DetectionConfig.LIDAR_BUFFER_SIZE:
                self.prev_points = points
                return

            # Merge point clouds from buffer
            merged_points = np.vstack(self.lidar_buffer)
            
            if self.prev_points is None or self.prev_points.shape[0] == 0:
                self.prev_points = merged_points
                return

            # Align point clouds
            aligned_prev, _ = PointCloudProcessor.align_point_clouds(
                self.prev_points, merged_points
            )

            # Detect dynamic points
            dynamic_points = self._detect_dynamic_points(aligned_prev, merged_points)
            
            # Cluster dynamic points
            cluster_centroids = self._cluster_dynamic_points(dynamic_points)
            
            # Update tracks
            self._update_tracks(cluster_centroids)
            
            # Visualize and publish results
            self._visualize_and_publish(msg.header, dynamic_points)
            
            self.prev_points = merged_points

        except Exception as e:
            self.logger.error(f"Error in lidar_callback: {str(e)}")

    def _detect_dynamic_points(self, aligned_prev: np.ndarray, current_points: np.ndarray) -> np.ndarray:
        """Detects dynamic points using distance threshold."""
        tree = cKDTree(aligned_prev)
        dists, _ = tree.query(current_points, k=1)
        dynamic_mask = dists > DetectionConfig.LIDAR_DYNAMIC_DIST_THRESHOLD
        return current_points[dynamic_mask]

    def _cluster_dynamic_points(self, dynamic_points: np.ndarray) -> np.ndarray:
        """Clusters dynamic points using DBSCAN."""
        if dynamic_points.shape[0] == 0:
            return np.array([])

        pcd_dyn = o3d.geometry.PointCloud()
        pcd_dyn.points = o3d.utility.Vector3dVector(dynamic_points)
        
        labels = np.array(pcd_dyn.cluster_dbscan(
            eps=DetectionConfig.CLUSTER_EPS,
            min_points=DetectionConfig.CLUSTER_MIN_POINTS,
            print_progress=False
        ))
        
        cluster_centroids = []
        for lbl in np.unique(labels[labels >= 0]):
            pts = dynamic_points[labels == lbl]
            centroid = np.mean(pts, axis=0)
            cluster_centroids.append(centroid)
            
        return np.array(cluster_centroids)

    def _update_tracks(self, cluster_centroids: np.ndarray) -> None:
        """Updates object tracks using Hungarian algorithm for assignment."""
        assigned, unassigned = self._match_clusters(cluster_centroids)
        new_tracks = {}

        # Update existing tracks
        for tid, cidx in assigned.items():
            centroid = cluster_centroids[cidx]
            track = self.tracks[tid]
            track.ukf.predict()
            track.ukf.update(centroid[:2])
            
            vx, vy = track.ukf.x[2], track.ukf.x[3]
            speed = np.linalg.norm([vx, vy])
            
            if tid not in self.motion_history:
                self.motion_history[tid] = deque(maxlen=DetectionConfig.MOTION_WINDOW_SIZE)
            
            self.motion_history[tid].append(int(speed > DetectionConfig.UKF_SPEED_THRESHOLD))
            is_dynamic = sum(self.motion_history[tid]) >= DetectionConfig.MOTION_COUNT_THRESHOLD
            
            new_tracks[tid] = TrackedObject(
                ukf=track.ukf,
                centroid=centroid,
                velocity=(float(vx), float(vy)),
                type='dynamic' if is_dynamic else 'static',
                last_update=time.time()
            )
            
            if is_dynamic:
                self.logger.info(
                    f"Dynamic object detected! id={tid}, "
                    f"pos={centroid[:2]}, speed={speed:.2f} m/s"
                )

        # Create new tracks for unassigned clusters
        for cidx in unassigned:
            centroid = cluster_centroids[cidx]
            tid = self.next_id
            self.next_id += 1
            
            ukf = SensorFusion.create_ukf()
            ukf.x[:2] = centroid[:2]
            
            self.motion_history[tid] = deque(maxlen=DetectionConfig.MOTION_WINDOW_SIZE)
            new_tracks[tid] = TrackedObject(
                ukf=ukf,
                centroid=centroid,
                velocity=(0.0, 0.0),
                type='static',
                last_update=time.time()
            )

        self.tracks = new_tracks

    def _match_clusters(self, cluster_centroids: np.ndarray) -> Tuple[Dict[int, int], List[int]]:
        """Matches clusters to existing tracks using Hungarian algorithm."""
        if not self.tracks or len(cluster_centroids) == 0:
            return {}, list(range(len(cluster_centroids)))

        from scipy.optimize import linear_sum_assignment
        
        track_ids = list(self.tracks.keys())
        cost = np.zeros((len(track_ids), len(cluster_centroids)), dtype=np.float32)
        
        for i, tid in enumerate(track_ids):
            prev_pos = self.tracks[tid].centroid[:2]
            for j, centroid in enumerate(cluster_centroids):
                cost[i, j] = np.linalg.norm(prev_pos - centroid[:2])
        
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {}
        unassigned = set(range(len(cluster_centroids)))
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < DetectionConfig.MAX_MATCH_DISTANCE:
                assignment[track_ids[r]] = c
                unassigned.discard(c)
        
        return assignment, list(unassigned)

    def _visualize_and_publish(self, header, dynamic_points: np.ndarray) -> None:
        """Visualizes and publishes detection results."""
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot ego vehicle
        ax.scatter(0, 0, c='red', marker='x', label='Ego Vehicle')
        
        # Plot dynamic points
        if dynamic_points.shape[0] > 0:
            ax.scatter(
                dynamic_points[:, 0],
                dynamic_points[:, 1],
                c='blue',
                s=5,
                alpha=0.5,
                label='Dynamic Points'
            )
        
        # Plot tracked objects
        dynamic_positions = []
        for track in self.tracks.values():
            if track.type == 'dynamic':
                dynamic_positions.append(track.centroid[:2])
        
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            ax.scatter(
                dynamic_positions[:, 0],
                dynamic_positions[:, 1],
                c='black',
                label='Tracked Objects'
            )
        
        # Configure plot
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Dynamic Object Detection')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticks(np.arange(-10, 11, 2))
        ax.set_yticks(np.arange(-10, 11, 2))
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Create and publish image message
        img_msg = Image()
        img_msg.header = header
        img_msg.height, img_msg.width, _ = img.shape
        img_msg.encoding = 'bgr8'
        img_msg.data = img.tobytes()
        img_msg.step = img_msg.width * 3
        
        self.pub.publish(img_msg)

def main():
    """Main entry point."""
    rclpy.init()
    node = LidarImuDynamicDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
