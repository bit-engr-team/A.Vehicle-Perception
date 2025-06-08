###!/usr/bin/env python3
import os
import cv2
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import json
import time
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# ---------------------- Ayarlar ----------------------
MODE = "test"  # "test" veya "live"  (argparse ile üzerine yazılacak)
FRAME_RATE = 20
MAX_MATCH_DISTANCE = 35
STATIC_CLASSES = [
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "bus stop", "street light", "traffic sign", "bicycle lane", "road",
    "sidewalk", "curb", "building", "wall", "fence", "sign", "kite"
]
UKF_TRACK_LIMIT = 30

# ---------------- Thresholds ----------------
UKF_SPEED_THRESHOLD = 0.3
MOTION_WINDOW_SIZE = 5
MOTION_COUNT_THRESHOLD = 3
FLOW_THRESHOLD = 1.0
LIDAR_DYNAMIC_DIST_THRESHOLD = 0.2
LIDAR_CLUSTER_TRACK_DIST = 2.0

# ---------------------- Test Modu Varsayılan Path’ler ----------------------
IMU_CSV_PATH = "input_data/exp27/sensor_abs_data.csv"
LIDAR_FOLDER = "input_data/exp27/lidar"
CAMERA_FOLDER = "input_data/exp27/camera"
OUTPUT_CSV = "output/test_tracking_output.csv"
OUTPUT_JSON = "output/test_tracking_output.json"

# ---------------------- YOLOv5 Model Yükleme ----------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4
# Eğer GPU kullanacaksanız:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# ---------------------- Logging ----------------------
import logging

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

# ---------------------- LiDAR → Görüntü Projeksiyonu ----------------------
def project_lidar_to_image(points: np.ndarray,
                           scale: float = 20,
                           offset_x: float = 640,
                           offset_y: float = 360) -> np.ndarray:
    cam_x = points[:, 0] * scale + offset_x
    cam_y = -points[:, 1] * scale + offset_y
    return np.vstack((cam_x, cam_y)).T

# ---------------------- k-NN Ortalama ile Gerçek Pozisyon ----------------------
def get_real_pos_from_bbox_center(lidar_proj: np.ndarray,
                                  lidar_points: np.ndarray,
                                  cx: float,
                                  cy: float,
                                  k: int = 3) -> np.ndarray:
    diffs = lidar_proj - np.array([cx, cy])
    dists = np.linalg.norm(diffs, axis=1)
    if dists.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    idxs = np.argsort(dists)[:k]
    pts = lidar_points[idxs]
    return np.mean(pts, axis=0)

# ---------------------- Nesne Tipi Güncelleme ----------------------
def update_object_type(obj: dict,
                       ukf_state: np.ndarray,
                       cls_name: str,
                       motion_history_deque: deque) -> None:
    if cls_name in STATIC_CLASSES:
        obj['type'] = "static"
    else:
        num_moving = sum(motion_history_deque)
        if num_moving >= MOTION_COUNT_THRESHOLD:
            obj['type'] = "dynamic"
        else:
            obj['type'] = "static"

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
                np.array(track['real_pos'][:2]) - np.array(det['real_pos'][:2])
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned = {}
    unassigned = set(range(M))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < MAX_MATCH_DISTANCE:
            assigned[track_ids[r]] = c
            unassigned.discard(c)

    return assigned, list(unassigned)

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

# ---------------------- FRAME İşleme ----------------------
def process_frame(
    frame: np.ndarray,
    lidar: np.ndarray,
    imu_accel: np.ndarray,
    object_tracks: dict,
    ukf_pool: dict,
    motion_history: dict,
    next_id: int
) -> (dict, int):
    lidar_proj = project_lidar_to_image(lidar)
    results = model(frame).xyxy[0].cpu().numpy()

    filtered_objs = []
    for det in results:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cls_name = model.names[int(cls)]
        real_pos = get_real_pos_from_bbox_center(lidar_proj, lidar, cx, cy, k=3)

        obj = {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": [float(cx), float(cy)],
            "real_pos": real_pos.tolist(),
            # Burada 2B yerine 3B norm alıyoruz:
            "distance": float(np.linalg.norm(real_pos)),
            "class": cls_name
        }
        filtered_objs.append(obj)

    assigned, unassigned = match_objects(object_tracks, filtered_objs)
    new_tracks = {}

    for track_id, det_idx in assigned.items():
        obj = filtered_objs[det_idx]
        if track_id not in ukf_pool:
            ukf_pool[track_id] = create_ukf()
            ukf_pool[track_id].x[:2] = np.array(obj['real_pos'][:2])
        ukf = ukf_pool[track_id]
        ukf.predict()
        ukf.update(np.array(obj['real_pos'][:2]))

        vx, vy = ukf.x[2], ukf.x[3]
        speed = np.linalg.norm([vx, vy])

        if track_id not in motion_history:
            motion_history[track_id] = deque(maxlen=MOTION_WINDOW_SIZE)
        motion_history[track_id].append(int(speed > UKF_SPEED_THRESHOLD))

        obj['velocity'] = [float(vx), float(vy)]
        obj['motion_count'] = sum(motion_history[track_id])
        obj['id'] = track_id

        update_object_type(obj, ukf.x, obj['class'], motion_history[track_id])
        new_tracks[track_id] = obj

    for det_idx in unassigned:
        obj = filtered_objs[det_idx]
        obj_id = next_id

        if len(ukf_pool) < UKF_TRACK_LIMIT:
            ukf_pool[obj_id] = create_ukf()
            ukf_pool[obj_id].x[:2] = np.array(obj['real_pos'][:2])
        motion_history[obj_id] = deque(maxlen=MOTION_WINDOW_SIZE)
        motion_history[obj_id].append(0)

        obj['velocity'] = [0.0, 0.0]
        obj['motion_count'] = 0
        obj['id'] = obj_id
        obj['type'] = "static"

        new_tracks[obj_id] = obj
        next_id += 1

    return new_tracks, next_id

# ---------------------- TEST MODU ----------------------
def run_test_mode():
    try:
        imu_data = pd.read_csv(IMU_CSV_PATH)
    except Exception as e:
        logger.error(f"IMU CSV okunamadı: {e}")
        return

    lidar_filenames = sorted(f for f in os.listdir(LIDAR_FOLDER) if f.endswith('.ply'))
    camera_filenames = sorted(f for f in os.listdir(CAMERA_FOLDER) if f.endswith('.png'))

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

    frames = []
    camera_timestamps = []
    logger.info("Kamera kareleri yükleniyor...")
    for fname in camera_filenames:
        path = os.path.join(CAMERA_FOLDER, fname)
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Kamera dosyası okunamadı ({fname}), atlandı.")
            continue
        frames.append(img)
        try:
            ts = float(os.path.splitext(fname)[0].split("_")[1])
        except:
            ts = time.time()
        camera_timestamps.append(ts)
    logger.info("Kamera yüklemesi tamamlandı.")

    object_tracks = {}
    ukf_pool = {}
    motion_history = {}
    next_id = 0
    last_positions = {}
    json_data = []

    prev_lidar = None
    prev_frame_gray = None
    prev_dynamic_centroids = np.empty((0, 3), dtype=np.float32)

    output_folder = os.path.dirname(OUTPUT_CSV)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # --- CSV Başlığını Güncelliyoruz: Sadece timestamp, id, x, y, z, distance ---
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    headers = ["timestamp", "id", "x", "y", "z", "distance"]
    csv_file.write(",".join(headers) + "\n")

    total_frames = len(frames)
    frame_interval = 1.0 / FRAME_RATE

    for idx in range(total_frames):
        start_time = time.time()

        frame = frames[idx]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lidar = cleaned_lidars[idx]
        timestamp = lidar_timestamps[idx]

        row = imu_data.iloc[(imu_data['timestamp'] - timestamp).abs().argsort()[:1]]
        if abs(row['timestamp'].values[0] - timestamp) > 0.3:
            continue
        imu_accel = row[['accel_x', 'accel_y']].values[0]

        if prev_frame_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame_gray, frame_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        else:
            flow = None

        if prev_lidar is not None:
            lidar_dynamic = detect_lidar_dynamic_clusters(prev_lidar, lidar)
            dynamic_centroids = lidar_dynamic["cluster_centroids"]
        else:
            dynamic_centroids = np.empty((0, 3), dtype=np.float32)

        object_tracks, next_id = process_frame(
            frame, lidar, imu_accel,
            object_tracks, ukf_pool, motion_history, next_id
        )

        for track_id, track in object_tracks.items():
            x1, y1, x2, y2 = track['bbox']

            cond_ukf = (track['motion_count'] >= MOTION_COUNT_THRESHOLD)

            cond_flow = False
            flow_mag = 0.0
            if flow is not None:
                fx = flow[y1:y2, x1:x2, 0]
                fy = flow[y1:y2, x1:x2, 1]
                if fx.size > 0 and fy.size > 0:
                    mag = np.sqrt(fx**2 + fy**2)
                    flow_mag = float(np.mean(mag))
                    if flow_mag > FLOW_THRESHOLD:
                        cond_flow = True

            cond_lidar = False
            if dynamic_centroids.shape[0] > 0:
                track_pos = np.array(track['real_pos'])
                dists_to_clusters = np.linalg.norm(dynamic_centroids[:, :2] - track_pos[:2], axis=1)
                if np.any(dists_to_clusters < LIDAR_CLUSTER_TRACK_DIST):
                    cond_lidar = True

            if track['class'] in STATIC_CLASSES:
                track['type'] = "static"
            else:
                if cond_ukf or cond_flow or cond_lidar:
                    track['type'] = "dynamic"
                else:
                    track['type'] = "static"

            track['flow_mag'] = flow_mag
            track['lidar_motion'] = cond_lidar

        for track_id, track in object_tracks.items():
            x1, y1, x2, y2 = track['bbox']

            is_pole = track['real_pos'][2] > 2.0

            if track_id in last_positions:
                prev_pos = np.array(last_positions[track_id])
                curr_pos = np.array(track['real_pos'])
                is_tilted = np.linalg.norm(curr_pos[:2] - prev_pos[:2]) > 0.2
            else:
                is_tilted = False

            last_positions[track_id] = track['real_pos']

            if track['type'] == "dynamic":
                color = (0, 255, 0)
            elif is_pole:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{track_id}:{track['type']}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if flow is not None:
                fx = flow[y1:y2, x1:x2, 0]
                fy = flow[y1:y2, x1:x2, 1]
                if fx.size > 0 and fy.size > 0:
                    avg_fx = np.mean(fx)
                    avg_fy = np.mean(fy)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.arrowedLine(frame, center,
                                    (int(center[0] + avg_fx * 5), int(center[1] + avg_fy * 5)),
                                    (255, 0, 0), 1, tipLength=0.3)

        # --- CSV ve JSON Çıktısını Güncelliyoruz ---
        # JSON katmanında frame düzeyinde "timestamp" tutulmaya devam ediyor.
        json_frame = {"timestamp": timestamp, "objects": []}
        for obj_id, obj in object_tracks.items():
            x, y, z = obj['real_pos']
            dist = obj['distance']

            # CSV: sadece time, id, x, y, z, distance
            csv_file.write(f"{timestamp},{obj_id},{x:.3f},{y:.3f},{z:.3f},{dist:.3f}\n")

            # JSON: sadece id, real_pos, distance
            json_frame["objects"].append({
                "id": obj_id,
                "real_pos": obj["real_pos"],
                "distance": dist
            })
        json_data.append(json_frame)

        cv2.imshow("Perception Test Mode (Final Dinamik Tespiti)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(max(0.0, frame_interval - elapsed))

        prev_frame_gray = frame_gray.copy()
        prev_lidar = lidar.copy()
        prev_dynamic_centroids = dynamic_centroids.copy()

    cv2.destroyAllWindows()
    csv_file.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info("Test modu (final) tamamlandı.")

# ---------------------- CANLI MOD ----------------------
def run_live_mode():
    import rospy
    from sensor_msgs.msg import PointCloud2, Imu, Image
    from cv_bridge import CvBridge
    import ros_numpy

    bridge = CvBridge()

    object_tracks = {}
    ukf_pool = {}
    motion_history = {}
    next_id = 0

    last_positions = {}
    json_data = []
    csv_file = None

    prev_lidar = None
    prev_frame_gray = None
    prev_dynamic_centroids = np.empty((0, 3), dtype=np.float32)
    prev_cam_timestamp = None
    prev_lidar_timestamp = None

    frame_interval = 1.0 / FRAME_RATE

    imu_accel = None
    latest_lidar = None
    latest_lidar_timestamp = None

    folder = os.path.dirname(OUTPUT_CSV)
    if folder:
        os.makedirs(folder, exist_ok=True)

    try:
        csv_file = open(OUTPUT_CSV, 'w', newline='')
    except Exception as e:
        logger.error(f"CSV dosyası açılamadı: {e}")
        return

    # --- CSV Başlığını Güncelliyoruz: Sadece timestamp, id, x, y, z, distance ---
    headers = ["timestamp", "id", "x", "y", "z", "distance"]
    csv_file.write(",".join(headers) + "\n")

    def imu_callback(msg):
        nonlocal imu_accel
        imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    def lidar_callback(msg):
        nonlocal latest_lidar, latest_lidar_timestamp
        try:
            points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        except Exception as e:
            logger.error(f"LiDAR msg alınamadı: {e}")
            return

        cleaned = clean_lidar(points, voxel_size=0.3, radius=1.2, min_neighbors=8)
        latest_lidar = cleaned
        latest_lidar_timestamp = msg.header.stamp.to_sec()

    def camera_callback(msg):
        nonlocal object_tracks, ukf_pool, motion_history, next_id
        nonlocal prev_lidar, prev_frame_gray, prev_dynamic_centroids
        nonlocal prev_cam_timestamp, prev_lidar_timestamp, imu_accel, latest_lidar
        nonlocal csv_file, json_data

        try:
            frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f"CVBridge çevirme hatası: {e}")
            return

        cam_timestamp = msg.header.stamp.to_sec()

        if latest_lidar is None or imu_accel is None:
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame_gray, frame_gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
            except Exception as e:
                logger.warning(f"Optik akış hesaplarken hata: {e}")
                flow = None
        else:
            flow = None

        if prev_lidar is not None:
            try:
                lidar_dynamic = detect_lidar_dynamic_clusters(prev_lidar, latest_lidar)
                dynamic_centroids = lidar_dynamic["cluster_centroids"]
            except Exception as e:
                logger.warning(f"LiDAR cluster tespiti hatası: {e}")
                dynamic_centroids = np.empty((0, 3), dtype=np.float32)
        else:
            dynamic_centroids = np.empty((0, 3), dtype=np.float32)

        object_tracks, next_id = process_frame(
            frame, latest_lidar, imu_accel,
            object_tracks, ukf_pool, motion_history, next_id
        )

        for track_id, track in object_tracks.items():
            x1, y1, x2, y2 = track['bbox']

            cond_ukf = (track['motion_count'] >= MOTION_COUNT_THRESHOLD)

            cond_flow = False
            flow_mag = 0.0
            if flow is not None:
                fx_crop = flow[y1:y2, x1:x2, 0]
                fy_crop = flow[y1:y2, x1:x2, 1]
                if fx_crop.size > 0 and fy_crop.size > 0:
                    mag = np.sqrt(fx_crop ** 2 + fy_crop ** 2)
                    flow_mag = float(np.mean(mag))
                    if flow_mag > FLOW_THRESHOLD:
                        cond_flow = True

            cond_lidar = False
            if dynamic_centroids.shape[0] > 0:
                track_pos = np.array(track['real_pos'])
                dists_to_clusters = np.linalg.norm(dynamic_centroids[:, :2] - track_pos[:2], axis=1)
                if np.any(dists_to_clusters < LIDAR_CLUSTER_TRACK_DIST):
                    cond_lidar = True

            if track['class'] in STATIC_CLASSES:
                track['type'] = "static"
            else:
                if cond_ukf or cond_flow or cond_lidar:
                    track['type'] = "dynamic"
                else:
                    track['type'] = "static"

            track['flow_mag'] = flow_mag
            track['lidar_motion'] = cond_lidar

        for track_id, track in object_tracks.items():
            x1, y1, x2, y2 = track['bbox']

            is_pole = (track['real_pos'][2] > 2.0)

            if track_id in last_positions:
                prev_pos = np.array(last_positions[track_id])
                curr_pos = np.array(track['real_pos'])
                is_tilted = (np.linalg.norm(curr_pos[:2] - prev_pos[:2]) > 0.2)
            else:
                is_tilted = False

            last_positions[track_id] = track['real_pos']

            if track['type'] == "dynamic":
                color = (0, 255, 0)
            elif is_pole:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{track_id}:{track['type']}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if flow is not None:
                fx_crop = flow[y1:y2, x1:x2, 0]
                fy_crop = flow[y1:y2, x1:x2, 1]
                if fx_crop.size > 0 and fy_crop.size > 0:
                    avg_fx = np.mean(fx_crop)
                    avg_fy = np.mean(fy_crop)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.arrowedLine(frame, center,
                                    (int(center[0] + avg_fx * 5), int(center[1] + avg_fy * 5)),
                                    (255, 0, 0), 1, tipLength=0.3)

        # --- CSV ve JSON Çıktısını Güncelliyoruz ---
        json_frame = {"timestamp": cam_timestamp, "objects": []}
        for obj_id, obj in object_tracks.items():
            x, y, z = obj['real_pos']
            dist = obj['distance']

            # CSV: sadece time, id, x, y, z, distance
            csv_file.write(f"{cam_timestamp},{obj_id},{x:.3f},{y:.3f},{z:.3f},{dist:.3f}\n")

            # JSON: sadece id, real_pos, distance
            json_frame["objects"].append({
                "id": obj_id,
                "real_pos": obj["real_pos"],
                "distance": dist
            })
        json_data.append(json_frame)

        cv2.imshow("Perception Live Mode (Final Dinamik Tespiti)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Canlı modda 'q' tuşuna basıldı, çıkılıyor.")
            rospy.signal_shutdown("User interrupted via 'q'")
            # CSV dosyasını kapatıp JSON’u yazıyoruz
            csv_file.close()
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(json_data, f, indent=2)
            return

        elapsed = time.time() - cam_timestamp

        prev_frame_gray = frame_gray.copy()
        prev_lidar = latest_lidar.copy()
        prev_dynamic_centroids = dynamic_centroids.copy()
        prev_cam_timestamp = cam_timestamp
        prev_lidar_timestamp = latest_lidar_timestamp

    # Döngü bittikten sonra kapatma
    csv_file.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info("Live modu (final) tamamlandı.")
    rospy.spin()

# ---------------------- Ana Fonksiyon ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perception system')
    parser.add_argument('--mode', choices=['test', 'live'], default='test',
                        help="Çalışma modu: 'test' (offline) veya 'live' (canlı)")
    parser.add_argument('--imu_csv', type=str, default=None,
                        help="(Opsiyonel) IMU CSV dosya yolu (test modunda)")
    parser.add_argument('--lidar_folder', type=str, default=None,
                        help="(Opsiyonel) LiDAR klasörü (test modunda)")
    parser.add_argument('--camera_folder', type=str, default=None,
                        help="(Opsiyonel) Kamera klasörü (test modunda)")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="(Opsiyonel) Çıktı CSV dosyası")
    parser.add_argument('--output_json', type=str, default=None,
                        help="(Opsiyonel) Çıktı JSON dosyası")
    args = parser.parse_args()

    if args.imu_csv:
        IMU_CSV_PATH = args.imu_csv
    if args.lidar_folder:
        LIDAR_FOLDER = args.lidar_folder
    if args.camera_folder:
        CAMERA_FOLDER = args.camera_folder
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
        import rospy
        rospy.init_node('perception_node', anonymous=True)
        run_live_mode()

    gc.collect()
