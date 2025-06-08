#!/usr/bin/env python3
"""
carla_dynamic_detector.py

A professional-ready script for real-time dynamic object detection and tracking
using LiDAR (and optional IMU) data in the CARLA simulator. Combines multi-frame
ICP alignment, DBSCAN clustering, UKF-based tracking, and live Matplotlib visualization.

Usage:
    python3 carla_dynamic_detector.py --host 127.0.0.1 --port 2000

"""
import argparse
import signal
import sys
import time
import logging

import carla
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt

# -------------------------- Parameters --------------------------
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 2000
LIDAR_DYNAMIC_DIST_THRESHOLD = 0.2
CLUSTER_EPS = 0.5
CLUSTER_MIN_POINTS = 5
MAX_MATCH_DISTANCE = 2.0
UKF_SPEED_THRESHOLD = 0.3
MOTION_WINDOW_SIZE = 5
MOTION_COUNT_THRESHOLD = 3
BUFFER_SIZE = 10  # number of frames to buffer for multi-frame ICP
DT = 1.0 / 10.0  # assumed LiDAR dt (s)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CarlaDynamicDetector")


def fx(x, dt):
    """State transition function for UKF: constant velocity model."""
    return np.array([
        x[0] + x[2] * dt,
        x[1] + x[3] * dt,
        x[2],
        x[3]
    ])


def hx(x):
    """Measurement function for UKF: position only."""
    return np.array([x[0], x[1]])


def create_ukf():
    """Initialize and return an Unscented Kalman Filter instance."""
    points = MerweScaledSigmaPoints(n=4, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=DT, points=points)
    ukf.x = np.zeros(4)
    ukf.P *= 10.0
    ukf.R = np.eye(2) * 0.5
    ukf.Q = np.diag([0.05, 0.05, 0.2, 0.2])
    return ukf


class CarlaDynamicDetector:
    """Main class to manage CARLA connection, sensors, detection, tracking, and visualization."""
    def __init__(self, host: str, port: int):
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()

        # Internal state
        self.lidar_buffer = deque(maxlen=BUFFER_SIZE)
        self.tracks = {}
        self.motion_history = {}
        self.next_id = 0

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-LIDAR_RANGE, LIDAR_RANGE)
        self.ax.set_ylim(-LIDAR_RANGE, LIDAR_RANGE)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Dynamic Objects')
        self.ax.plot(0, 0, 'rx', label='Origin')
        self.raw_scatter = self.ax.scatter([], [], s=2, alpha=0.4, label='Raw Dynamic')
        self.centroid_scatter = self.ax.scatter([], [], c='k', label='Centroids')
        self.ax.legend()

        # Spawn actors
        self._spawn_vehicle_and_sensors()
        # Register signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _spawn_vehicle_and_sensors(self):
        # Spawn vehicle
        veh_bp = self.blueprints.filter('vehicle.*')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(veh_bp, spawn_point)

        # LIDAR sensor
        lidar_bp = self.blueprints.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('channels', '32')
        lidar_transform = carla.Transform(carla.Location(z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(self._lidar_callback)

        # Optional IMU sensor (not yet fused in processing)
        imu_bp = self.blueprints.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(z=2.5))
        self.imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        self.imu.listen(lambda msg: None)

        logger.info("Spawned vehicle and sensors.")

    def _lidar_callback(self, msg):
        """Handle incoming LiDAR frames, process and visualize."""
        raw = np.frombuffer(msg.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        pts = self._clean_lidar(raw)
        if pts.size == 0:
            return

        # Buffer and require at least two frames
        self.lidar_buffer.append(pts)
        if len(self.lidar_buffer) < 2:
            return

        # Multi-frame merge
        merged = np.vstack(self.lidar_buffer)
        prev_all = np.vstack(list(self.lidar_buffer)[:-1])

        # ICP alignment
        reg = o3d.pipelines.registration.registration_icp(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prev_all)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged)),
            1.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        aligned_prev = np.asarray(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prev_all)).transform(reg.transformation).points
        )

        # Extract dynamic points
        tree = cKDTree(aligned_prev)
        dists, _ = tree.query(merged, k=1)
        dyn_pts = merged[dists > LIDAR_DYNAMIC_DIST_THRESHOLD]

        # Cluster and track
        centroids = self._cluster_centroids(dyn_pts)
        self._associate_and_update(dyn_pts, centroids)
        self._update_visualization(dyn_pts, centroids)

    def _clean_lidar(self, points: np.ndarray) -> np.ndarray:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(0.3)
        pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=1.2)
        arr = np.asarray(pcd.points)
        return arr[arr[:,2] > 0.2]

    def _cluster_centroids(self, dyn_pts: np.ndarray) -> np.ndarray:
        if dyn_pts.size == 0:
            return np.empty((0, 3))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dyn_pts))
        labels = np.array(pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=False))
        centroids = [dyn_pts[labels==lbl].mean(axis=0) for lbl in np.unique(labels[labels>=0])]
        return np.array(centroids)

    def _associate_and_update(self, dyn_pts: np.ndarray, centroids: np.ndarray):
        # Build cost matrix
        ids = list(self.tracks.keys())
        cost = np.zeros((len(ids), len(centroids)))
        for i, tid in enumerate(ids):
            prev = self.tracks[tid]['centroid'][:2]
            for j, c in enumerate(centroids):
                cost[i,j] = np.linalg.norm(prev - c[:2])

        from scipy.optimize import linear_sum_assignment
        assignment, unassigned = {}, list(range(len(centroids)))
        if cost.size > 0:
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                if cost[ri,ci] < MAX_MATCH_DISTANCE:
                    assignment[ids[ri]] = ci
                    unassigned.remove(ci)

        new_tracks = {}
        # Update existing
        for tid, idx in assignment.items():
            cen = centroids[idx]
            ukf = self.tracks[tid]['ukf']
            ukf.predict(); ukf.update(cen[:2])
            speed = np.linalg.norm(ukf.x[2:])
            hist = self.motion_history.setdefault(tid, deque(maxlen=MOTION_WINDOW_SIZE))
            hist.append(int(speed > UKF_SPEED_THRESHOLD))
            is_dyn = sum(hist) >= MOTION_COUNT_THRESHOLD
            new_tracks[tid] = {'ukf':ukf, 'centroid':cen, 'type':'dynamic' if is_dyn else 'static'}

        # Add new
        for idx in unassigned:
            tid = self.next_id; self.next_id+=1
            ukf = create_ukf(); ukf.x[:2] = centroids[idx][:2]
            self.motion_history[tid] = deque(maxlen=MOTION_WINDOW_SIZE)
            new_tracks[tid] = {'ukf':ukf, 'centroid':centroids[idx], 'type':'static'}

        self.tracks = new_tracks

    def _update_visualization(self, dyn_pts: np.ndarray, centroids: np.ndarray):
        self.raw_scatter.set_offsets(dyn_pts[:,:2])
        dyn_centroids = np.array([t['centroid'][:2] for t in self.tracks.values() if t['type']=='dynamic'])
        self.centroid_scatter.set_offsets(dyn_centroids if dyn_centroids.size else np.empty((0,2)))
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        # Stop sensors
        if self.lidar: self.lidar.stop()
        if self.imu:   self.imu.stop()
        # Destroy actors
        for actor in [self.lidar, self.imu, self.vehicle]:
            if actor: actor.destroy()
        plt.ioff(); plt.close(self.fig)
        logger.info("Clean shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Dynamic Detector with UKF Tracking")
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help='CARLA host')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='CARLA port')
    return parser.parse_args()


def main():
    args = parse_args()
    detector = CarlaDynamicDetector(args.host, args.port)
    logger.info("Starting detection loop. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        detector.shutdown()


if __name__ == '__main__':
    main()
