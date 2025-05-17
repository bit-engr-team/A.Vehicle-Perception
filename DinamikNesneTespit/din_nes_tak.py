# robotaxi_perception_final.py - test/live modlu ROS 2 uyumlu perception sistemi

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import json
import time
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment

MODE = "test"  # "test" veya "live"

if MODE == "live":
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu, PointCloud2, Image
    from cv_bridge import CvBridge
    import sensor_msgs.point_cloud2 as pc2

FRAME_RATE = 20
MAX_MATCH_DISTANCE = 2.5
STATIC_CLASSES = [
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "bus stop", "street light", "traffic sign", "bicycle lane", "road",
    "sidewalk", "curb", "building", "wall", "fence", "sign", "kite"
]
UKF_TRACK_LIMIT = 30

if MODE == "test":
    IMU_CSV_PATH = r"C:\Users\esahi\Downloads\exp27\exp27\sensor_abs_data.csv"
    LIDAR_FOLDER = r"C:\Users\esahi\Downloads\exp27\exp27\lidar"
    CAMERA_FOLDER = r"C:\Users\esahi\Downloads\exp27\exp27\camera"
    OUTPUT_CSV = "output/test_tracking_output.csv"
    OUTPUT_JSON = "output/test_tracking_output.json"
else:
    OUTPUT_CSV = "output/live_output.csv"
    OUTPUT_JSON = "output/live_output.json"

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.3

def fx(x, dt):
    return np.array([
        x[0] + x[2]*dt + 0.5*x[4]*dt**2,
        x[1] + x[3]*dt + 0.5*x[5]*dt**2,
        x[2], x[3], x[4], x[5]
    ])

def hx(x):
    return np.array([x[0], x[1]])

def create_ukf():
    points = MerweScaledSigmaPoints(n=6, alpha=1e-3, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=2, fx=fx, hx=hx, dt=1.0/FRAME_RATE, points=points)
    ukf.x = np.zeros(6)
    ukf.P *= 10
    ukf.R = np.eye(2) * 0.1
    return ukf

def clean_lidar(points):
    points = np.asarray(points)
    mask = points[:, 2] > 0.2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered = np.asarray(pcd.points)
    return filtered[np.linalg.norm(filtered[:, :2], axis=1) < 50.0]

def project_lidar_to_image(points, scale=20, offset_x=640, offset_y=360):
    cam_x = points[:, 0] * scale + offset_x
    cam_y = -points[:, 1] * scale + offset_y
    return np.vstack((cam_x, cam_y)).T

def update_object_type(obj, velocity, cls_name):
    if cls_name in STATIC_CLASSES:
        obj['type'] = "static"
    else:
        speed = np.linalg.norm(velocity)
        obj['type'] = "dynamic" if speed > 0.5 else "static"
def match_objects(tracks, detections):
    if not tracks or not detections:
        return {}, list(range(len(detections)))
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks.values()):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = np.linalg.norm(np.array(track['real_pos'][:2]) - np.array(det['real_pos'][:2]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned, unassigned = {}, set(range(len(detections)))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < MAX_MATCH_DISTANCE:
            assigned[list(tracks.keys())[r]] = c
            unassigned.discard(c)
    return assigned, list(unassigned)

def process_frame(frame, lidar, imu_accel, object_tracks, ukf_pool, next_id):
    imu_vel = imu_accel * FRAME_RATE
    lidar_proj = project_lidar_to_image(lidar)
    results = model(frame).xyxy[0].cpu().numpy()
    filtered_objs, used_indices = [], set()
    for det in results:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cls_name = model.names[int(cls)]
        dist = np.linalg.norm(lidar_proj - np.array([cx, cy]), axis=1)
        closest_idx = np.argmin(dist)
        if closest_idx in used_indices or closest_idx >= len(lidar): continue
        used_indices.add(closest_idx)
        real_pos = lidar[closest_idx]
        obj = {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": [float(cx), float(cy)],
            "real_pos": real_pos.tolist(),
            "distance": float(np.linalg.norm(real_pos[:2])),
            "velocity": imu_vel.tolist(),
            "class": cls_name
        }
        update_object_type(obj, imu_vel, cls_name)
        filtered_objs.append(obj)
    for i, point in enumerate(lidar):
        if i not in used_indices:
            obj = {
                "bbox": None,
                "center": None,
                "real_pos": point.tolist(),
                "distance": float(np.linalg.norm(point[:2])),
                "velocity": imu_vel.tolist(),
                "class": "unknown"
            }
            update_object_type(obj, imu_vel, "unknown")
            filtered_objs.append(obj)
    assigned, unassigned = match_objects(object_tracks, filtered_objs)
    new_tracks = {}
    for track_id, det_idx in assigned.items():
        obj = filtered_objs[det_idx]
        new_tracks[track_id] = obj
        if track_id in ukf_pool:
            ukf_pool[track_id].predict()
            ukf_pool[track_id].update(np.array(obj['real_pos'][:2]))
        else:
            if len(ukf_pool) < UKF_TRACK_LIMIT:
                ukf_pool[track_id] = create_ukf()
    for det_idx in unassigned:
        obj = filtered_objs[det_idx]
        obj['id'] = next_id
        new_tracks[next_id] = obj
        if len(ukf_pool) < UKF_TRACK_LIMIT:
            ukf_pool[next_id] = create_ukf()
        next_id += 1
    return new_tracks, next_id

def run_test_mode():
    imu_data = pd.read_csv(IMU_CSV_PATH)
    lidar_files = sorted(f for f in os.listdir(LIDAR_FOLDER) if f.endswith('.ply'))
    camera_files = sorted(f for f in os.listdir(CAMERA_FOLDER) if f.endswith('.png'))
    object_tracks = {}
    ukf_pool = {}
    next_id = 0
    json_data = []
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    csv_file = open(OUTPUT_CSV, 'w')
    csv_file.write("timestamp,x,y,z,distance,type\n")
    for idx in range(len(camera_files)):
        frame = cv2.imread(os.path.join(CAMERA_FOLDER, camera_files[idx]))
        lidar_path = os.path.join(LIDAR_FOLDER, lidar_files[idx])
        try:
            lidar = clean_lidar(np.asarray(o3d.io.read_point_cloud(lidar_path).points))
        except:
            continue
        timestamp = float(os.path.splitext(lidar_files[idx].split('_')[1])[0])
        row = imu_data.iloc[(imu_data['timestamp'] - timestamp).abs().argsort()[:1]]
        if abs(row['timestamp'].values[0] - timestamp) > 0.3:
            continue
        imu_accel = row[['accel_x', 'accel_y']].values[0]
        object_tracks, next_id = process_frame(frame, lidar, imu_accel, object_tracks, ukf_pool, next_id)
        json_frame = {"timestamp": timestamp, "objects": []}
        for obj_id, obj in object_tracks.items():
            csv_file.write(f"{timestamp},{obj['real_pos'][0]},{obj['real_pos'][1]},{obj['real_pos'][2]},{obj['distance']},{obj['type']}\n")
            json_frame["objects"].append({
                "real_pos": obj["real_pos"],
                "type": obj["type"],
                "distance": obj["distance"]
            })
        json_data.append(json_frame)
        cv2.imshow("Perception Test Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    csv_file.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
    print("Test mode complete.")

def run_live_mode():
    import rospy
    from sensor_msgs.msg import PointCloud2, Imu, Image
    from cv_bridge import CvBridge
    bridge = CvBridge()
    object_tracks = {}
    ukf_pool = {}
    next_id = 0
    csv_file = open(OUTPUT_CSV, 'w')
    csv_file.write("timestamp,x,y,z,distance,type\n")

    imu_accel = None
    latest_lidar = None

    def imu_callback(msg):
        nonlocal imu_accel
        imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])

    def lidar_callback(msg):
        nonlocal latest_lidar
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        latest_lidar = clean_lidar(pc)

    def camera_callback(msg):
        nonlocal object_tracks, next_id, ukf_pool, imu_accel
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        if latest_lidar is None or imu_accel is None:
            return
        object_tracks, next_id = process_frame(frame, latest_lidar, imu_accel, object_tracks, ukf_pool, next_id)
        timestamp = msg.header.stamp.to_sec()
        for obj in object_tracks.values():
            csv_file.write(f"{timestamp},{obj['real_pos'][0]},{obj['real_pos'][1]},{obj['real_pos'][2]},{obj['distance']},{obj['type']}\n")
        csv_file.flush()

    rospy.init_node('perception_node', anonymous=True)
    rospy.Subscriber('/imu/data', Imu, imu_callback)
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback)
    rospy.Subscriber('/camera/image_raw', Image, camera_callback)

    try:
        rospy.spin()  # BURADA PROGRAM ASLA KENDİ KAPANMAZ
    except KeyboardInterrupt:
        print("Live mode stopped by user.")
    finally:
        csv_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perception system')
    parser.add_argument('--mode', choices=['test', 'live'], default='test')
    args = parser.parse_args()
    if args.mode == 'test':
        run_test_mode()
    else:
        run_live_mode()


import gc
gc.collect()
