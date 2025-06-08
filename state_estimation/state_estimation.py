#!/usr/bin/env python3
"""
State estimation with a 6-state Kalman filter:
- State: [lat, lon, v_lat, v_lon, a_lat, a_lon]
- Constant-acceleration model
- GNSS for position/altitude
- IMU for acceleration
- Velocity initialized from first 5 GNSS points
- Inputs: new_data/gnss.csv, new_data/imu.csv
- Output: new_data/processed_sensor_data.csv
"""
import csv
import numpy as np
from filterpy.kalman import KalmanFilter

GNSS_CSV = 'new_data/gnss.csv'
IMU_CSV = 'new_data/imu.csv'
OUTPUT_CSV = 'new_data/processed_sensor_data.csv'

# Outlier threshold for IMU acceleration (m/s^2)
IMU_ACC_THRESHOLD = 1000

class StateEstimator:
    def __init__(self, initial_lat, initial_lon, initial_v_lat, initial_v_lon):
        self.kf = self.init_kalman_filter(initial_lat, initial_lon, initial_v_lat, initial_v_lon)

    def init_kalman_filter(self, lat, lon, v_lat, v_lon):
        kf = KalmanFilter(dim_x=6, dim_z=2)
        # State: [lat, lon, v_lat, v_lon, a_lat, a_lon]
        kf.x = np.array([lat, lon, v_lat, v_lon, 0, 0], dtype=float)
        # F: constant-acceleration model
        kf.F = np.eye(6)
        # These will be updated each step with dt
        # H: measure position only
        kf.H = np.zeros((2, 6))
        kf.H[0, 0] = 1  # lat
        kf.H[1, 1] = 1  # lon
        # Initial uncertainty
        kf.P *= 1
        # GPS measurement noise (tune as needed)
        kf.R = np.eye(2) * 2e-6
        # Process noise (tune as needed)
        q = 1e-8
        kf.Q = np.eye(6) * q
        return kf

    def process_row(self, gnss_row, imu_row, dt):
        # Update F for current dt (constant-acceleration model)
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * dt ** 2
        F[1, 5] = 0.5 * dt ** 2
        F[2, 4] = dt
        F[3, 5] = dt
        self.kf.F = F
        # Predict step: use IMU for acceleration
        try:
            acc_lat = float(imu_row['accel_x'])
            acc_lon = float(imu_row['accel_y'])
        except Exception:
            acc_lat = 0
            acc_lon = 0
        if abs(acc_lat) > IMU_ACC_THRESHOLD or abs(acc_lon) > IMU_ACC_THRESHOLD:
            acc_lat = 0
            acc_lon = 0
        acc_lat *= 1e-5
        acc_lon *= 1e-5
        # Set acceleration in state for prediction
        self.kf.x[4] = acc_lat
        self.kf.x[5] = acc_lon
        self.kf.predict()
        # GNSS update
        lat = float(gnss_row['lat'])
        lon = float(gnss_row['lon'])
        z = np.array([lat, lon])
        self.kf.update(z)
        # Output
        lat, lon, v_lat, v_lon, a_lat, a_lon = self.kf.x
        return {
            'timestamp': float(gnss_row['timestamp']),
            'lat': float(lat),
            'lon': float(lon),
            'altitude': float(gnss_row['alt']),
            'velocity_lat': float(v_lat),
            'velocity_lon': float(v_lon),
            'accel_lat': float(a_lat),
            'accel_lon': float(a_lon)
        }

def estimate_initial_velocity(rows, n=5):
    # Use the first n rows to estimate initial velocity from GNSS
    if len(rows) < 2:
        return 0, 0
    lats = [float(row['lat']) for row in rows[:n]]
    lons = [float(row['lon']) for row in rows[:n]]
    times = [float(row['timestamp']) for row in rows[:n]]
    v_lats = []
    v_lons = []
    for i in range(1, len(lats)):
        dt = times[i] - times[i-1]
        if dt > 0:
            v_lats.append((lats[i] - lats[i-1]) / dt)
            v_lons.append((lons[i] - lons[i-1]) / dt)
    v_lat = np.median(v_lats) if v_lats else 0
    v_lon = np.median(v_lons) if v_lons else 0
    return v_lat, v_lon

def load_csv_as_list(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def find_nearest_imu(imu_rows, target_time, last_idx):
    # Find the IMU row with timestamp closest to target_time, starting from last_idx
    best_idx = last_idx
    best_dt = abs(float(imu_rows[last_idx]['timestamp']) - target_time)
    for i in range(last_idx, min(last_idx+10, len(imu_rows))):
        dt = abs(float(imu_rows[i]['timestamp']) - target_time)
        if dt < best_dt:
            best_dt = dt
            best_idx = i
        else:
            break
    return imu_rows[best_idx], best_idx

def main():
    gnss_rows = load_csv_as_list(GNSS_CSV)
    imu_rows = load_csv_as_list(IMU_CSV)
    if not gnss_rows or not imu_rows:
        print('No data!')
        return
    first_row = gnss_rows[0]
    initial_lat = float(first_row['lat'])
    initial_lon = float(first_row['lon'])
    initial_v_lat, initial_v_lon = estimate_initial_velocity(gnss_rows, n=5)
    estimator = StateEstimator(initial_lat, initial_lon, initial_v_lat, initial_v_lon)
    with open(OUTPUT_CSV, 'w', newline='') as outfile:
        fieldnames = ['timestamp', 'lat', 'lon', 'altitude', 'velocity_lat', 'velocity_lon', 'accel_lat', 'accel_lon']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        prev_time = float(first_row['timestamp'])
        imu_idx = 0
        for gnss_row in gnss_rows:
            curr_time = float(gnss_row['timestamp'])
            dt = max(0.01, curr_time - prev_time)
            imu_row, imu_idx = find_nearest_imu(imu_rows, curr_time, imu_idx)
            result = estimator.process_row(gnss_row, imu_row, dt)
            writer.writerow(result)
            prev_time = curr_time
    print(f"Processed data written to {OUTPUT_CSV}")

# Note: Install dependencies with:
# pip install numpy filterpy

if __name__ == '__main__':
    main()
