import pandas as pd
import os


# Read the input data
df = pd.read_csv('exp-ler/2.csv')

# Convert GNSS data
gnss_df = df[['timestamp', 'latitude', 'longitude', 'altitude']].copy()
gnss_df.columns = ['timestamp', 'latitude', 'longitude', 'altitude']
gnss_df.to_csv('exp-ler/2-gnss.csv', index=False)

# Convert IMU data
imu_df = df[['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].copy()
imu_df.to_csv('exp-ler/2-imu.csv', index=False)

print("Conversion completed successfully!") 