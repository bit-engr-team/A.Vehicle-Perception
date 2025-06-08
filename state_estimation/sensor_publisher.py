# carla_sensor_server.py
import carla
import time
import json
import base64
import numpy as np
import logging
import csv
import os
from carla_base import CarlaSimManagerBase

HOST = "localhost"
PORT = 2000
OUTPUT_DIR = "new_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sensör konfigürasyonları
SENSOR_CONFIGURATIONS = {
    "front_camera": {
        "blueprint": "sensor.camera.rgb",
        "transform": carla.Transform(carla.Location(z=1, x=1),carla.Rotation(yaw = 0)),
        "config": {"image_size_x": "640", 
                   "image_size_y": "640", 
                   "fov": "120",
                   "sensor_tick":"0.1"
                   },
        "callback": "camera_front_callback"
    },
    "sensor_lidar": {
        "blueprint": "sensor.lidar.ray_cast",
        "transform": carla.Transform(carla.Location(z=1,x=1)),
        "config": {"channels": "16", 
                   "points_per_second": "300000", 
                   "range": "100", 
                   "rotation_frequency": "10",
                   "upper_fov" : "10",
                   "lower_fov" : "-10",
                   #"dropoff_general_rate":"0",
                   #"dropoff_intensity_limit":"0",
                   #"dropoff_zero_intensity":"0",
                   "sensor_tick":"0.1"
                   },
        "callback": "lidar_callback"
    },
    "gnss_sensor": {
        "blueprint": "sensor.other.gnss",
        "transform": carla.Transform(),
        "config": {
            "noise_alt_bias": "0.5",           # Yükseklik için 0.5 m bias
            "noise_alt_stddev": "0.5",         # Yükseklik için 1.5 m stddev

            "noise_lat_bias":   "0.000001",     # Enlem için küçük bias (~0.1 m)
            "noise_lat_stddev": "0.000003",     # Enlem için ~0.3 m stddev

            "noise_lon_bias":  "-0.000001",     # Boylam için küçük bias (~0.1 m)
            "noise_lon_stddev": "0.000003",     # Boylam için ~0.3 m stddev

            "noise_seed": "42",
            "sensor_tick": "0.02"
            },
        "callback": "gnss_callback"
    },

    "imu_sensor": {
        "blueprint": "sensor.other.imu",
        "transform": carla.Transform(),
        "config": {
            "noise_accel_stddev_x": "0.03",   # m/s^2
            "noise_accel_stddev_y": "0.03",
            "noise_accel_stddev_z": "0.03",

            "noise_gyro_bias_x":   "0.0002",  # rad/s
            "noise_gyro_bias_y":   "0.0002",
            "noise_gyro_bias_z":   "0.0002",

            "noise_gyro_stddev_x": "0.0005",  # rad/s
            "noise_gyro_stddev_y": "0.0005",
            "noise_gyro_stddev_z": "0.0005",

            "noise_seed": "42",
            "sensor_tick": "0.02"
        },
        "callback": "imu_callback"
    }
}

class CarlaSensorLogger(CarlaSimManagerBase):
    def __init__(self, client):
        super().__init__(client)
        # Open CSV files for each sensor
        self.gnss_file = open(os.path.join(OUTPUT_DIR, 'gnss.csv'), 'w', newline='')
        self.imu_file = open(os.path.join(OUTPUT_DIR, 'imu.csv'), 'w', newline='')
        self.lidar_file = open(os.path.join(OUTPUT_DIR, 'lidar.bin'), 'wb')
        self.camera_file = open(os.path.join(OUTPUT_DIR, 'camera.bin'), 'wb')
        # Write headers
        self.gnss_writer = csv.DictWriter(self.gnss_file, fieldnames=['timestamp', 'lat', 'lon', 'alt'])
        self.gnss_writer.writeheader()
        self.imu_writer = csv.DictWriter(self.imu_file, fieldnames=[
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z'])
        self.imu_writer.writeheader()

    def gnss_callback(self, data):
        self.gnss_writer.writerow({
            'timestamp': data.timestamp,
            'lat': data.latitude,
            'lon': data.longitude,
            'alt': data.altitude
        })
        self.gnss_file.flush()

    def imu_callback(self, data):
        self.imu_writer.writerow({
            'timestamp': getattr(data, "timestamp", time.time()),
            'accel_x': data.accelerometer.x,
            'accel_y': data.accelerometer.y,
            'accel_z': data.accelerometer.z,
            'gyro_x': data.gyroscope.x,
            'gyro_y': data.gyroscope.y,
            'gyro_z': data.gyroscope.z
        })
        self.imu_file.flush()

    def camera_front_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        jpg_bytes = array[:, :, :3].copy()
        self.camera_file.write(jpg_bytes.tobytes())
        self.camera_file.flush()

    def lidar_callback(self, lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        points[:, 0] *= -1
        self.lidar_file.write(points.astype(np.float32).tobytes())
        self.lidar_file.flush()

    def destroy_sensors(self):
        super().destroy_sensors()
        self.gnss_file.close()
        self.imu_file.close()
        self.lidar_file.close()
        self.camera_file.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    client.load_world("Town01_Opt")
    logging.info(f"CARLA istemcisine bağlanıldı: {HOST}:{PORT}")
    logger = CarlaSensorLogger(client)
    ego_vehicle = logger.spawn_vehicle("vehicle.micro.microlino", 0)
    if ego_vehicle is None:
        logging.error("Araç oluşturulamadı.")
        return
    callback_map = {
        "gnss_callback": logger.gnss_callback,
        "imu_callback": logger.imu_callback,
        "camera_front_callback": logger.camera_front_callback,
        "lidar_callback": logger.lidar_callback,
    }
    logger.setup_sensors(ego_vehicle, SENSOR_CONFIGURATIONS, callback_map)
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    ego_vehicle.set_autopilot(True, tm_port)
    tm.global_percentage_speed_difference(75)
    logger.start_all_sensors()
    print("Tüm sensör dinleyicileri başlatıldı.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Durduruluyor...")
    finally:
        logger.stop_all_sensors()
        logger.destroy_sensors()

if __name__ == '__main__':
    main()