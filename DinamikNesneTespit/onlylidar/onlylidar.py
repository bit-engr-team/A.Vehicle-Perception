####!/usr/bin/env python3
import os
import numpy as np
import open3d as o3d
import pandas as pd
import json
import time
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import logging

# ---------------------- Ayarlar ----------------------
MODE = "test"  # "test" veya "live"
FRAME_RATE = 20
MAX_MATCH_DISTANCE = 35
UKF_TRACK_LIMIT = 30

# ---------------- Thresholds ----------------
UKF_SPEED_THRESHOLD = 0.3
MOTION_WINDOW_SIZE = 5
MOTION_COUNT_THRESHOLD = 3
LIDAR_DYNAMIC_DIST_THRESHOLD = 0.2
LIDAR_CLUSTER_TRACK_DIST = 2.0

# ---------------------- Test Modu Varsayılan Path’ler ----------------------
IMU_CSV_PATH = r"D:\bit\data\exp27\sensor_abs_data.csv"
LIDAR_FOLDER = r"D:\bit\data\exp27\lidar"
OUTPUT_CSV = r"C:\Users\esahi\OneDrive\Masaüstü\bittt\lidar.csv"
OUTPUT_JSON = r"C:\Users\esahi\OneDrive\Masaüstü\bittt\lidar.json"

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("perception.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- UKF Fonksiyonları ----------------------
def fx(x: np.ndarray, dt: float) -> np.ndarray:
    return np.array([
        x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
        x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
        x[2], x[3], x[4], x[5]
    ])

def hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], x[1]])

def create_ukf() -> UnscentedKalmanFilter:
    points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=1.0/FRAME_RATE, points=points)
    ukf.x = np.zeros(6)
    ukf.P *= 10
    ukf.R = np.eye(2) * 0.5
    ukf.Q = np.diag([0.05, 0.05, 0.2, 0.2, 0.5, 0.5])
    return ukf

# ---------------------- LiDAR Temizleme ----------------------
def clean_lidar(points: np.ndarray,
                voxel_size: float = 0.3,
                radius: float = 1.2,
                min_neighbors: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    filtered = np.asarray(pcd.points, dtype=np.float32)
    mask = filtered[:, 2] > 0.2
    filtered = filtered[mask]
    dists_xy = np.linalg.norm(filtered[:, :2], axis=1)
    return filtered[dists_xy < 50.0]

# ---------------------- LiDAR Dinamik Cluster Tespiti ----------------------
def detect_lidar_dynamic_clusters(prev_points: np.ndarray,
                                  curr_points: np.ndarray) -> dict:
    pcd_prev = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(prev_points)
    pcd_prev_ds = pcd_prev.voxel_down_sample(voxel_size=0.5)

    pcd_curr = o3d.geometry.PointCloud()
    pcd_curr.points = o3d.utility.Vector3dVector(curr_points)
    pcd_curr_ds = pcd_curr.voxel_down_sample(voxel_size=0.5)

    threshold = 1.0
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_prev_ds, pcd_curr_ds, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation

    pcd_prev_full = o3d.geometry.PointCloud()
    pcd_prev_full.points = o3d.utility.Vector3dVector(prev_points)
    pcd_prev_full.transform(transformation)
    aligned_prev = np.asarray(pcd_prev_full.points)

    tree = cKDTree(aligned_prev[:, :3])
    dists, _ = tree.query(curr_points[:, :3], k=1)
    dynamic_mask = dists > LIDAR_DYNAMIC_DIST_THRESHOLD
    dynamic_points = curr_points[dynamic_mask]

    cluster_centroids = np.empty((0, 3), dtype=np.float32)
    if dynamic_points.shape[0] > 0:
        pcd_dyn = o3d.geometry.PointCloud()
        pcd_dyn.points = o3d.utility.Vector3dVector(dynamic_points)
        labels = np.array(
            pcd_dyn.cluster_dbscan(eps=0.5, min_points=5, print_progress=False)
        )
        unique_labels = np.unique(labels[labels >= 0])
        for lbl in unique_labels:
            cluster_pts = dynamic_points[labels == lbl]
            centroid = np.mean(cluster_pts, axis=0)
            cluster_centroids = np.vstack([cluster_centroids, centroid])
    return {"cluster_centroids": cluster_centroids}

# ---------------------- Nesne Eşleme (Hungarian) ----------------------
def match_objects(tracks: dict, detections: list) -> (dict, list):
    if not tracks or not detections:
        return {}, list(range(len(detections)))
    N = len(tracks)
    M = len(detections)
    cost_matrix = np.zeros((N, M), dtype=np.float32)
    track_ids = list(tracks.keys())
    for i, tid in enumerate(track_ids):
        track = tracks[tid]
        for j, det in enumerate(detections):
            cost_matrix[i, j] = np.linalg.norm(
                np.array(track['center'][:2]) - np.array(det['center'][:2])
            )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned = {}
    unassigned = set(range(M))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < MAX_MATCH_DISTANCE:
            assigned[track_ids[r]] = c
            unassigned.discard(c)
    return assigned, list(unassigned)

# ---------------------- LiDAR Kümeleme ve Takip (Frame İşleme) ----------------------
def process_frame_lidar(
    lidar: np.ndarray,
    imu_accel: np.ndarray,
    object_tracks: dict,
    ukf_pool: dict,
    motion_history: dict,
    next_id: int
) -> (dict, int, list):
    # DBSCAN ile LiDAR küme merkezi bul
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar)
    labels = np.array(pcd.cluster_dbscan(eps=1.2, min_points=8, print_progress=False))
    unique_labels = np.unique(labels[labels >= 0])
    detections = []
    for lbl in unique_labels:
        cluster_pts = lidar[labels == lbl]
        center = np.mean(cluster_pts, axis=0)
        detections.append({
            "center": center.tolist(),
        })

    assigned, unassigned = match_objects(object_tracks, detections)
    new_tracks = {}

    for track_id, det_idx in assigned.items():
        det = detections[det_idx]
        if track_id not in ukf_pool:
            ukf_pool[track_id] = create_ukf()
            ukf_pool[track_id].x[:2] = np.array(det['center'][:2])
        ukf = ukf_pool[track_id]
        ukf.predict()
        ukf.update(np.array(det['center'][:2]))
        vx, vy = ukf.x[2], ukf.x[3]
        speed = np.linalg.norm([vx, vy])
        if track_id not in motion_history:
            motion_history[track_id] = deque(maxlen=MOTION_WINDOW_SIZE)
        motion_history[track_id].append(int(speed > UKF_SPEED_THRESHOLD))
        det['velocity'] = [float(vx), float(vy)]
        det['motion_count'] = sum(motion_history[track_id])
        det['id'] = track_id
        det['type'] = "dynamic" if sum(motion_history[track_id]) >= MOTION_COUNT_THRESHOLD else "static"
        new_tracks[track_id] = det

    for det_idx in unassigned:
        det = detections[det_idx]
        obj_id = next_id
        if len(ukf_pool) < UKF_TRACK_LIMIT:
            ukf_pool[obj_id] = create_ukf()
            ukf_pool[obj_id].x[:2] = np.array(det['center'][:2])
        motion_history[obj_id] = deque(maxlen=MOTION_WINDOW_SIZE)
        motion_history[obj_id].append(0)
        det['velocity'] = [0.0, 0.0]
        det['motion_count'] = 0
        det['id'] = obj_id
        det['type'] = "static"
        new_tracks[obj_id] = det
        next_id += 1

    return new_tracks, next_id, detections

# ---------------------- TEST MODU ----------------------
def run_test_mode():
    try:
        imu_data = pd.read_csv(IMU_CSV_PATH)
    except Exception as e:
        logger.error(f"IMU CSV okunamadı: {e}")
        return

    lidar_filenames = sorted(f for f in os.listdir(LIDAR_FOLDER) if f.endswith('.ply'))
    cleaned_lidars = []
    lidar_timestamps = []
    logger.info("LiDAR verileri yükleniyor ve temizleniyor...")
    for fname in lidar_filenames:
        path = os.path.join(LIDAR_FOLDER, fname)
        try:
            raw_pcd = np.asarray(o3d.io.read_point_cloud(path).points)
        except Exception as e:
            logger.error(f"LiDAR dosyası okunamadı ({fname}): {e}")
            continue
        cleaned = clean_lidar(raw_pcd, voxel_size=0.3, radius=1.2, min_neighbors=8)
        cleaned_lidars.append(cleaned)
        try:
            ts = float(os.path.splitext(fname)[0].split("_")[1])
        except:
            ts = time.time()
        lidar_timestamps.append(ts)
    logger.info("LiDAR yüklemesi tamamlandı.")

    object_tracks = {}
    ukf_pool = {}
    motion_history = {}
    next_id = 0
    json_data = []

    prev_lidar = None
    prev_dynamic_centroids = np.empty((0, 3), dtype=np.float32)

    output_folder = os.path.dirname(OUTPUT_CSV)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    csv_file = open(OUTPUT_CSV, 'w', newline='')
    headers = ["timestamp", "id", "x", "y", "z", "distance"]
    csv_file.write(",".join(headers) + "\n")

    total_frames = len(cleaned_lidars)
    frame_interval = 1.0 / FRAME_RATE

    for idx in range(total_frames):
        start_time = time.time()
        lidar = cleaned_lidars[idx]
        timestamp = lidar_timestamps[idx]
        row = imu_data.iloc[(imu_data['timestamp'] - timestamp).abs().argsort()[:1]]
        if abs(row['timestamp'].values[0] - timestamp) > 0.3:
            continue
        imu_accel = row[['accel_x', 'accel_y']].values[0]

        if prev_lidar is not None:
            lidar_dynamic = detect_lidar_dynamic_clusters(prev_lidar, lidar)
            dynamic_centroids = lidar_dynamic["cluster_centroids"]
        else:
            dynamic_centroids = np.empty((0, 3), dtype=np.float32)

        object_tracks, next_id, detections = process_frame_lidar(
            lidar, imu_accel,
            object_tracks, ukf_pool, motion_history, next_id
        )

        # --- CSV ve JSON Çıktısını Güncelliyoruz ---
        json_frame = {"timestamp": timestamp, "objects": []}
        for obj_id, obj in object_tracks.items():
            x, y, z = obj['center']
            dist = np.linalg.norm([x, y, z])
            csv_file.write(f"{timestamp},{obj_id},{x:.3f},{y:.3f},{z:.3f},{dist:.3f}\n")
            json_frame["objects"].append({
                "id": obj_id,
                "real_pos": [x, y, z],
                "distance": dist
            })
        json_data.append(json_frame)

        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(max(0.0, frame_interval - elapsed))
        prev_lidar = lidar.copy()
        prev_dynamic_centroids = dynamic_centroids.copy()

    csv_file.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info("Test modu (LiDAR-only) tamamlandı.")

# ---------------------- CANLI MOD ----------------------
def run_live_mode():
    import rospy
    from sensor_msgs.msg import PointCloud2, Imu
    import ros_numpy

    object_tracks = {}
    ukf_pool = {}
    motion_history = {}
    next_id = 0

    prev_lidar = None
    prev_dynamic_centroids = np.empty((0, 3), dtype=np.float32)

    frame_interval = 1.0 / FRAME_RATE
    imu_accel = None
    latest_lidar = None
    latest_lidar_timestamp = None

    folder = os.path.dirname(OUTPUT_CSV)
    if folder:
        os.makedirs(folder, exist_ok=True)
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    headers = ["timestamp", "id", "x", "y", "z", "distance"]
    csv_file.write(",".join(headers) + "\n")

    json_data = []

    def imu_callback(msg):
        nonlocal imu_accel
        imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    def lidar_callback(msg):
        nonlocal latest_lidar, latest_lidar_timestamp, object_tracks, ukf_pool, motion_history, next_id, prev_lidar, prev_dynamic_centroids, imu_accel, csv_file, json_data

        try:
            points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        except Exception as e:
            logger.error(f"LiDAR msg alınamadı: {e}")
            return
        cleaned = clean_lidar(points, voxel_size=0.3, radius=1.2, min_neighbors=8)
        timestamp = msg.header.stamp.to_sec()

        if imu_accel is None:
            return

        if prev_lidar is not None:
            lidar_dynamic = detect_lidar_dynamic_clusters(prev_lidar, cleaned)
            dynamic_centroids = lidar_dynamic["cluster_centroids"]
        else:
            dynamic_centroids = np.empty((0, 3), dtype=np.float32)

        object_tracks, next_id, detections = process_frame_lidar(
            cleaned, imu_accel,
            object_tracks, ukf_pool, motion_history, next_id
        )

        # --- CSV ve JSON Çıktısını Güncelliyoruz ---
        json_frame = {"timestamp": timestamp, "objects": []}
        for obj_id, obj in object_tracks.items():
            x, y, z = obj['center']
            dist = np.linalg.norm([x, y, z])
            csv_file.write(f"{timestamp},{obj_id},{x:.3f},{y:.3f},{z:.3f},{dist:.3f}\n")
            json_frame["objects"].append({
                "id": obj_id,
                "real_pos": [x, y, z],
                "distance": dist
            })
        json_data.append(json_frame)

        prev_lidar = cleaned.copy()
        prev_dynamic_centroids = dynamic_centroids.copy()

    rospy.init_node('perception_lidar_node', anonymous=True)
    rospy.Subscriber('/imu/data', Imu, imu_callback, queue_size=1)
    rospy.Subscriber('/lidar/points', PointCloud2, lidar_callback, queue_size=1)
    rospy.spin()
    csv_file.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info("Live modu (LiDAR-only) tamamlandı.")

# ---------------------- Ana Fonksiyon ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LiDAR-only Perception system')
    parser.add_argument('--mode', choices=['test', 'live'], default='test',
                        help="Çalışma modu: 'test' (offline) veya 'live' (canlı)")
    parser.add_argument('--imu_csv', type=str, default=None,
                        help="IMU CSV dosya yolu (test modunda)")
    parser.add_argument('--lidar_folder', type=str, default=None,
                        help="LiDAR klasörü (test modunda)")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="Çıktı CSV dosyası")
    parser.add_argument('--output_json', type=str, default=None,
                        help="Çıktı JSON dosyası")
    args = parser.parse_args()

    if args.imu_csv:
        IMU_CSV_PATH = args.imu_csv
    if args.lidar_folder:
        LIDAR_FOLDER = args.lidar_folder
    if args.output_csv:
        OUTPUT_CSV = args.output_csv
    if args.output_json:
        OUTPUT_JSON = args.output_json

    MODE = args.mode
    logger.info(f"Çalışma modu: {MODE}")

    import gc
    if MODE == 'test':
        run_test_mode()
    else:
        run_live_mode()
    gc.collect()
