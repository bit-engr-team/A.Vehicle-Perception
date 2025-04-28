import os
import cv2
import numpy as np
import open3d as o3d
import torch
import pandas as pd
import json
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ------------------ Ayarlar ------------------ #
IMU_CSV_PATH = r"D:\bit\exp26\exp26\sensor_abs_data.csv"
LIDAR_FOLDER = r"D:\bit\exp26\exp26\lidar"
CAMERA_FOLDER = r"D:\bit\exp26\exp26\camera"
OUTPUT_CSV = r"C:\Users\esahi\Downloads\tracking_output.csv"
OUTPUT_JSON = r"C:\Users\esahi\Downloads\tracking_output.json"
FRAME_RATE = 10
MAX_MATCH_DISTANCE = 50

# ------------------ YOLOv5 Modeli (CPU) ------------------ #
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.to('cpu')
model.conf = 0.4

# ------------------ UKF Fonksiyonları ------------------ #
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

# ------------------ Yardımcı Fonksiyonlar ------------------ #
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

def clean_lidar(points, z_threshold=0.2, nb_neighbors=20, std_ratio=2.0, max_range=50.0):
    mask = points[:, 2] > z_threshold
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered = np.asarray(pcd.points)
    return filtered[np.linalg.norm(filtered[:, :2], axis=1) < max_range]

def project_lidar_to_image(points, scale=20, offset_x=640, offset_y=360):
    cam_x = points[:, 0] * scale + offset_x
    cam_y = -points[:, 1] * scale + offset_y
    return np.vstack((cam_x, cam_y)).T

def match_objects(prev_tracks, current_objs, max_distance):
    if not prev_tracks or not current_objs:
        return {}, list(range(len(current_objs)))

    prev_centers = np.array([v['center'] for v in prev_tracks.values()])
    curr_centers = np.array([obj['center'] for obj in current_objs])
    cost_matrix = np.linalg.norm(prev_centers[:, None] - curr_centers[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned, unassigned = {}, set(range(len(current_objs)))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_distance:
            assigned[list(prev_tracks.keys())[r]] = c
            unassigned.discard(c)
    return assigned, list(unassigned)

# ------------------ Yardımcı Fonksiyon: Hız Eşik Kontrolü ------------------ #
def update_object_type(obj, velocity, speed_threshold=1.0):
    """
    Bağıl hız hesaplamasına göre nesnenin tipini günceller.
    Hız eşik değerini aşarsa dinamik, aksi takdirde statik olarak etiketler.
    """
    speed = np.linalg.norm(velocity)  # Nesnenin hızını hesapla (bağıl hız)
    
    if speed > speed_threshold:  # Bağıl hız eşik değerini kontrol et
        obj['type'] = "Dynamic"  # Eğer hız eşik değerinden büyükse, nesne dinamik olarak etiketlenir
    else:
        obj['type'] = "Static"   # Eğer hız eşik değerinden küçükse, nesne statik olarak etiketlenir

# ------------------ Başlangıç ------------------ #
imu_data = read_imu_data(IMU_CSV_PATH)
lidar_files = sorted(f for f in os.listdir(LIDAR_FOLDER) if f.endswith('.ply'))
camera_files = sorted(f for f in os.listdir(CAMERA_FOLDER) if f.endswith('.png'))

# Takip verileri
csv_file = open(OUTPUT_CSV, 'w')
csv_file.write("object_id,frame,x,y,vx,vy,ax,ay,distance,type,class\n")
json_data = []

object_tracks = {}
next_id = 0

STATIC_CLASSES = [
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "bus stop", "street light", "traffic sign", "bicycle lane", "road", "sidewalk",
    "curb", "building", "wall", "fence", "sign"
]

# ------------------ Ana Döngü ------------------ #
for idx in range(len(camera_files)):
    print(f"[Frame {idx+1}/{len(camera_files)}]")

    frame = cv2.imread(os.path.join(CAMERA_FOLDER, camera_files[idx]))
    if frame is None:
        print("[!] Görüntü okunamadı.")
        continue

    lidar_path = os.path.join(LIDAR_FOLDER, lidar_files[idx])
    try:
        lidar = clean_lidar(read_lidar(lidar_path))
        lidar_proj = project_lidar_to_image(lidar)
    except:
        continue

    timestamp = float(os.path.splitext(lidar_files[idx].split('_')[1])[0])
    imu_accel = get_closest_imu(imu_data, timestamp)
    if imu_accel is None:
        continue
    imu_vel = imu_accel * FRAME_RATE

    results = model(frame).xyxy[0].cpu().numpy()
    current_objs = []

    for det in results:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if lidar_proj.shape[0] == 0:
            continue
        dist = np.linalg.norm(lidar_proj - np.array([cx, cy]), axis=1)
        closest_idx = np.argmin(dist)
        real_pos = lidar_proj[closest_idx] / 20.0
        cls_name = model.names[int(cls)]

        # Statik mi Dinamik mi?
        if cls_name in STATIC_CLASSES:
            typ = "Static"
            color = (0, 0, 255)  # Kırmızı: Statik nesneler
        else:
            typ = "Dynamic"
            color = (0, 255, 0)  # Yeşil: Dinamik nesneler

        current_objs.append({
         "bbox": (int(x1), int(y1), int(x2), int(y2)),
         "center": np.array([cx, cy]),
         "real_pos": real_pos,
         "distance": np.linalg.norm(real_pos),
         "object_type": cls_name,
         "predicted_type": "Static" if cls_name in STATIC_CLASSES else "Dynamic",  # Sadece tahmin
         "velocity": imu_vel,       
        })

    assigned, unassigned = match_objects(object_tracks, current_objs, MAX_MATCH_DISTANCE)
    new_tracks = {}

    for track_id, obj_idx in assigned.items():
        obj = current_objs[obj_idx]
        ukf = object_tracks[track_id]['ukf']
        ukf.predict()
        ukf.update(obj["real_pos"])
        obj['ukf'] = ukf
        obj['velocity'] = ukf.x[2:4]
        update_object_type(obj, obj['velocity'])  # Bağıl hıza göre nesnenin tipini güncelle
        new_tracks[track_id] = obj

    for obj_idx in unassigned:
        obj = current_objs[obj_idx]
        ukf = create_ukf()
        ukf.x[:2] = obj['real_pos']
        obj['ukf'] = ukf
        obj['type'] = "Dynamic"
        new_tracks[next_id] = obj
        next_id += 1

    object_tracks = new_tracks

    for obj_id, obj in object_tracks.items():
        x1, y1, x2, y2 = obj["bbox"]
        vx, vy = obj["velocity"]
        pos = obj["real_pos"]
        dist = obj["distance"]
        typ = obj["type"]
        cls_name = obj["object_type"]

        csv_file.write(f"{obj_id},{idx},{pos[0]:.2f},{pos[1]:.2f},{vx:.2f},{vy:.2f},0,0,{dist:.2f},{typ},{cls_name}\n")

        json_data.append({
            "frame": idx,
            "id": obj_id,
            "class": cls_name,
            "type": typ,
            "position": pos.tolist(),
            "velocity": [vx, vy],
            "distance": float(dist)
        })

        # UKF sonrası güncel tipe göre renk seç
        draw_color = (0, 0, 255) if typ == "Static" else (0, 255, 0)

        label = f"ID:{obj_id} {cls_name} {typ}"
        info = f"Pos:[{pos[0]:.1f},{pos[1]:.1f}] Vel:[{vx:.1f},{vy:.1f}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        cv2.putText(frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

    cv2.imshow("Sensor Fusion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ Kapatma ------------------ #
csv_file.close()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w") as jf:
    json.dump(json_data, jf, indent=4)

print("✅ Takip tamamlandı. CSV ve JSON dosyaları kaydedildi.")
