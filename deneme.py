import carla
import random
import time
import matplotlib.pyplot as plt
import numpy as np                      #lidar kısmında kullanılıyor
import open3d as o3d                    # "
import os
import plotly.graph_objs as go
import numpy as np                               # "
import pandas as pd


gnss_abs_data = []
gnss_data = []
imu_abs_data = []
imu_data = []

host = "localhost"
port = 2000
client = carla.Client(host, port)

try:
    client.load_world("Town01_Opt")
except:
    pass
time.sleep(2)#async olmalı normalde




world = client.get_world()
level = world.get_map()


blueprint_library = world.get_blueprint_library()

#region Araba
vehicle_bp = blueprint_library.find("vehicle.micro.microlino")
spawn_points = world.get_map().get_spawn_points() #spawnpointlere bakılıyor
araba = world.spawn_actor(vehicle_bp, spawn_points[0]) #araba (actor) oluşturma komutu

# araba.set_autopilot(True)
# time.sleep(5)


#region Camera
#https://carla.readthedocs.io/en/0.9.15/ref_sensors/#rgb-camera
camera_transform = carla.Transform(carla.Location(x=0.6,z=1.3),carla.Rotation())
camera_bp = blueprint_library.find('sensor.camera.rgb')

camera_bp.set_attribute('sensor_tick', str(0.1))
camera_bp.set_attribute("fov", str(105) )  #fov 105
#camera_bp.set_attribute("fov", str(90) ) #fov 90
camera_bp.set_attribute("image_size_x", str(640) )
camera_bp.set_attribute("image_size_y", str(640) )

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=araba)

def camera_listen(image):
    # if tişme%10 == 0 ?? #saniyede 10 tane olacak en az
    image.save_to_disk(f'output/images/{image.timestamp}.png')
    return

#camera.stop()
#endregion

#region Lidar
#https://carla.readthedocs.io/en/0.9.15/python_api/#carla.SensorData
#https://carla.readthedocs.io/en/0.9.15/ref_sensors/#lidar-sensor

lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_transform = carla.Transform(carla.Location(x=0.5,z=1.3),carla.Rotation())

#Lidar parametreleri
lidar_bp.set_attribute('sensor_tick', str(0.1)) # Simulation seconds between sensor captures (ticks).
lidar_bp.set_attribute('rotation_frequency', str(10)) # Simulation seconds between sensor captures (ticks).
lidar_bp.set_attribute("channels", "16")
lidar_bp.set_attribute("lower_fov", "-10")



lidar = world.spawn_actor(lidar_bp,lidar_transform, attach_to=araba)

cloudX = []
cloudY = []
cloudZ = []

def read_lidar(data):
    #print(data)
    for point in data:
        if point.point.z > -1.2 and point.point.z < 0.8:  #buradaki sayılar lidarın z'sine göre -1.2 mesela
            cloudX.append(point.point.x)
            cloudY.append(point.point.y)
            cloudZ.append(point.point.z)


#lidar.stop()

#region IMU
imu_t = carla.Transform( carla.Location(), carla.Rotation())    #directly center
imu_bp = blueprint_library.find("sensor.other.imu")

imu_bp.set_attribute('sensor_tick', str(0.02))

imu_abs = world.spawn_actor(imu_bp,imu_t, attach_to = araba)

accel_bias = 0.05
imu_bp.set_attribute('noise_accel_stddev_x', str(accel_bias))
imu_bp.set_attribute('noise_accel_stddev_y', str(accel_bias))
imu_bp.set_attribute('noise_accel_stddev_z', str(accel_bias))

gyro_bias = 0.05
imu_bp.set_attribute('noise_gyro_bias_x', str(gyro_bias))
imu_bp.set_attribute('noise_gyro_bias_y', str(gyro_bias))
imu_bp.set_attribute('noise_gyro_bias_z', str(gyro_bias))

gyro_stddev = 0.05
imu_bp.set_attribute('noise_gyro_stddev_x', str(gyro_stddev))
imu_bp.set_attribute('noise_gyro_stddev_y', str(gyro_stddev))
imu_bp.set_attribute('noise_gyro_stddev_z', str(gyro_stddev))

imu = world.spawn_actor(imu_bp,imu_t, attach_to = araba)

def imu_callback(data):
    imu_data.append({
        'timestamp': data.timestamp,
        'accel_x': data.accelerometer.x,
        'accel_y': data.accelerometer.y,
        'accel_z': data.accelerometer.z,
        'gyro_x': data.gyroscope.x,
        'gyro_y': data.gyroscope.y,
        'gyro_z': data.gyroscope.z
    })

def imu_abs_callback(data):
    imu_abs_data.append({
        'timestamp': data.timestamp,
        'accel_x': data.accelerometer.x,
        'accel_y': data.accelerometer.y,
        'accel_z': data.accelerometer.z,
        'gyro_x': data.gyroscope.x,
        'gyro_y': data.gyroscope.y,
        'gyro_z': data.gyroscope.z
    })

#region GNSS
gnss_t = carla.Transform( carla.Location(), carla.Rotation())   #directly center
gnss_bp = blueprint_library.find("sensor.other.gnss")
gnss_bp.set_attribute('sensor_tick', str(0.02))

gnss_abs = world.spawn_actor(gnss_bp,gnss_t, attach_to = araba)

gnss_bias = 0.000005
gnss_bp.set_attribute('noise_alt_bias', str(gnss_bias))
gnss_bp.set_attribute('noise_lat_bias', str(gnss_bias))
gnss_bp.set_attribute('noise_lon_bias', str(gnss_bias))

gnss_stddev = 0.000022
gnss_bp.set_attribute('noise_alt_stddev', str(gnss_stddev))
gnss_bp.set_attribute('noise_lat_stddev', str(gnss_stddev))
gnss_bp.set_attribute('noise_lon_stddev', str(gnss_stddev))




gnss = world.spawn_actor(gnss_bp,gnss_t, attach_to = araba)




def gnss_abs_callback(data):
    gnss_abs_data.append({
        'timestamp': data.timestamp,
        'abs-latitude': data.latitude,
        'abs-longitude': data.longitude,
        'abs-altitude': data.altitude
    })

def gnss_callback(data):
    gnss_data.append({
        'timestamp': data.timestamp,
        'latitude': data.latitude,
        'longitude': data.longitude,
        'altitude': data.altitude
    })

#region Kayıt
spectator = world.get_spectator()
transform = araba.get_transform()
spectator.set_transform(carla.Transform(transform.location + camera_transform.location,
carla.Rotation(yaw=90)))

araba.set_autopilot(True)
#world.traffic_generator(n=40,w=40)
time.sleep(5)


#GPT İLE YAPILAN KISIM BASLANGIÇ
os.makedirs("output", exist_ok=True)

exp = len(os.listdir("output"))

output_dir = f"output/exp{exp+1}"
lidar_dir = output_dir+"/lidar"
camera_dir = output_dir+"/camera"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)
os.makedirs(camera_dir, exist_ok=True)


def save_lidar_data(point_cloud):
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    data = np.reshape(data, (len(data) // 4, 4))  # Reshape to (x, y, z, intensity)
    
    # Save as PLY file
    ply_filename = os.path.join(lidar_dir, f"L_{point_cloud.timestamp:05.5f}.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # Extract only (x, y, z)
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Saved: {ply_filename}")

    cloudX = []
    cloudY = []
    cloudZ = []

    for point in point_cloud:
        if point.point.z > -1.2 and point.point.z < 0.8:  #buradaki sayılar lidarın z'sine göre -1.2 mesela
            cloudX.append(point.point.x)
            cloudY.append(point.point.y)
            cloudZ.append(point.point.z)

def save_camera_data(image):
    # if tişme%10 == 0 ?? #saniyede 10 tane olacak en az
    image.save_to_disk(f'{camera_dir}/C_{image.timestamp:05.5f}.png')  #1407.3688582187751
    return
#GPT SON

#lidar.listen(lambda data: read_lidar(data))
lidar.listen(lambda point_cloud: save_lidar_data(point_cloud))
camera.listen(lambda image: save_camera_data(image))#kamera kayıt

gnss_abs.listen(lambda data: gnss_abs_callback(data))
gnss.listen(lambda data: gnss_callback(data))

imu_abs.listen(lambda data: imu_abs_callback(data))
imu.listen(lambda data: imu_callback(data))


#SENSÖRLERİN KAYIT SÜRESİ
time.sleep(10)

gnss_abs.stop()
gnss.stop()

imu.stop()
imu_abs.stop()

lidar.stop()
camera.stop()


print("File Save, Ömer bana karı al")
#GNSS IMU SAVE
gnss_abs_df = pd.DataFrame(gnss_abs_data)
gnss_df = pd.DataFrame(gnss_data)

imu_df = pd.DataFrame(imu_data)
imu_abs_df = pd.DataFrame(imu_abs_data)

merged_df = pd.merge_asof(gnss_df, imu_df, on='timestamp', direction='nearest')
merged_abs_df = pd.merge_asof(gnss_abs_df, imu_abs_df, on='timestamp', direction='nearest')

#merged_df.head()
merged_df.to_csv(f"{output_dir}/sensor_data.csv",index=False)
merged_abs_df.to_csv(f"{output_dir}/sensor_abs_data.csv",index=False)


time.sleep(2)
araba.destroy()




