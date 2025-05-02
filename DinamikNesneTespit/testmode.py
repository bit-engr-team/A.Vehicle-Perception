import os
import cv2
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import json
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment

########################## MOD SEÇİMİ ##########################
MODE = "test"  # "test" or "live"

########################## AYARLAR ##########################
FRAME_RATE = 10
MAX_MATCH_DISTANCE = 50
STATIC_CLASSES = ["traffic light", "stop sign", "fire hydrant", "parking meter", "bench", "bus stop",
                  "street light", "traffic sign", "bicycle lane", "road", "sidewalk", "curb",
                  "building", "wall", "fence", "sign"]

if MODE == "test":
    IMU_CSV_PATH = r"C:\\Users\\esahi\\Downloads\\yeni-lidar1\\exp9\\sensor_abs_data.csv"
    LIDAR_FOLDER = r"D:\\bit\\exp9\\lidar"
    CAMERA_FOLDER = r"D:\\bit\\exp9\\camera"
    OUTPUT_CSV = r"C:\\Users\\esahi\\Downloads\\tracking_output.csv"
    OUTPUT_JSON = r"C:\\Users\\esahi\\Downloads\\tracking_output.json"
else:
    OUTPUT_CSV = "live_output.csv"
    OUTPUT_JSON = "live_output.json"

########################## MODEL ##########################
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model.to('cpu')
model.conf = 0.4

########################## UKF ##########################
def fx(x, dt):
    return np.array([x[0] + x[2]*dt + 0.5*x[4]*dt**2,
                     x[1] + x[3]*dt + 0.5*x[5]*dt**2,
                     x[2], x[3], x[4], x[5]])

def hx(x):
    return np.array([x[0], x[1]])

def create_ukf():
    points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=1.0/FRAME_RATE, points=points)
    ukf.x = np.zeros(6)
    ukf.P *= 10
    ukf.R = np.eye(2) * 0.1
    return ukf

########################## HELPER ##########################
def read_imu_data(path):
    return pd.read_csv(path)

def get_closest_imu(imu_data, timestamp, threshold=0.2):
    row = imu_data.iloc[(imu_data['timestamp'] - timestamp).abs().argsort()[:1]]
    if abs(row['timestamp'].values[0] - timestamp) > threshold:
        return None
    return row[['accel_x', 'accel_y']].values[0]

def read_lidar(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def clean_lidar(points):
    mask = points[:, 2] > 0.2  # 2 cm threshold for ground height
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered = np.asarray(pcd.points)
    return filtered[np.linalg.norm(filtered[:, :2], axis=1) < 50.0]

def project_lidar_to_image(points, scale=20, offset_x=640, offset_y=360):
    cam_x = points[:, 0] * scale + offset_x
    cam_y = -points[:, 1] * scale + offset_y
    return np.vstack((cam_x, cam_y)).T

def is_pole(obj, height_threshold=2.0):
    return obj['real_pos'][2] > height_threshold

def detect_tilt(last_pos, current_pos, tilt_threshold=0.2):
    displacement = np.linalg.norm(current_pos - last_pos)
    return displacement > tilt_threshold

def filter_detections(results):
    filtered_results = []
    for det in results:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        cls_name = model.names[int(cls)]
        filtered_results.append(det)
    return filtered_results

def match_objects(tracks, detections):
    if not tracks or not detections:
        return {}, list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks.values()):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = np.linalg.norm(track['real_pos'][:2] - det['real_pos'][:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned, unassigned = {}, set(range(len(detections)))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < MAX_MATCH_DISTANCE:
            assigned[list(tracks.keys())[r]] = c
            unassigned.discard(c)
    return assigned, list(unassigned)

def update_object_type(obj, velocity):
    speed = np.linalg.norm(velocity)
    obj['type'] = "moving" if speed > 0.5 else "static"

########################## MAIN ##########################
ignore_classes = ["road", "sidewalk", "curb", "building", "wall", "fence", "sign"]

if MODE == "test":
    imu_data = read_imu_data(IMU_CSV_PATH)
    lidar_files = sorted(f for f in os.listdir(LIDAR_FOLDER) if f.endswith('.ply'))
    camera_files = sorted(f for f in os.listdir(CAMERA_FOLDER) if f.endswith('.png'))

csv_file = open(OUTPUT_CSV, 'w')
csv_file.write("object_id,frame,x,y,vx,vy,ax,ay,distance,type,class\n")
json_data = []

object_tracks = {}
next_id = 0
last_positions = {}

for idx in range(len(camera_files)):
    frame = cv2.imread(os.path.join(CAMERA_FOLDER, camera_files[idx]))
    lidar_path = os.path.join(LIDAR_FOLDER, lidar_files[idx])
    try:
        lidar = clean_lidar(read_lidar(lidar_path))
    except:
        continue
    timestamp = float(os.path.splitext(lidar_files[idx].split('_')[1])[0])
    imu_accel = get_closest_imu(imu_data, timestamp)
    if imu_accel is None:
        continue
    imu_vel = imu_accel * FRAME_RATE

    lidar_proj = project_lidar_to_image(lidar)
    results = model(frame).xyxy[0].cpu().numpy()
    filtered_results = filter_detections(results)

    current_objs = []
    for det in filtered_results:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if lidar_proj.shape[0] == 0:
            continue
        dist = np.linalg.norm(lidar_proj - np.array([cx, cy]), axis=1)
        closest_idx = np.argmin(dist)
        real_pos = lidar[closest_idx] if closest_idx < len(lidar) else np.array([0.0, 0.0, 0.0])

        obj = {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": np.array([cx, cy]),
            "real_pos": real_pos,
            "distance": np.linalg.norm(real_pos[:2]),
            "velocity": imu_vel,
            "type": "unknown",  # Initialize type here
        }

        obj['is_pole'] = is_pole(obj)

        obj_id = f"tmp_{idx}_{cx:.1f}_{cy:.1f}"
        if obj_id in last_positions:
            obj['is_tilted'] = detect_tilt(last_positions[obj_id], obj['real_pos'])
        else:
            obj['is_tilted'] = False
        last_positions[obj_id] = obj['real_pos']

        # Update the object type
        update_object_type(obj, imu_vel)

        current_objs.append(obj)

    assigned, unassigned = match_objects(object_tracks, current_objs)
    new_tracks = {}
    for track_id, detection_idx in assigned.items():
        track = object_tracks[track_id]
        detection = current_objs[detection_idx]
        track['real_pos'] = detection['real_pos']
        track['type'] = detection['type']  # Make sure the type is updated
        new_tracks[track_id] = track

    for idx in unassigned:
        new_tracks[next_id] = current_objs[idx]
        next_id += 1

    object_tracks = new_tracks
    for track_id, track in object_tracks.items():
        x1, y1, x2, y2 = track['bbox']
        if track['is_tilted']:
            color = (0, 255, 0)  # Moving object (Green)
        elif track['is_pole']:
            color = (0, 255, 255)  # Pole (Yellow)
        else:
            color = (0, 0, 255)  # Static object (Red)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Tracking", frame)
    cv2.waitKey(1)

    json_data.append(object_tracks)
    csv_file.write(f"{next_id},{idx},{track['real_pos'][0]},{track['real_pos'][1]},{track['velocity'][0]},{track['velocity'][1]},0,0,{track['distance']},{track['type']},{track['is_pole']}\n")

cv2.destroyAllWindows()
csv_file.close()
with open(OUTPUT_JSON, 'w') as json_out:
    json.dump(json_data, json_out)

print("Tracking completed.")
