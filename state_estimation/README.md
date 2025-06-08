# State Estimation Script

## Overview
`state_estimation.py` performs sensor fusion using a 6-state Kalman filter to estimate vehicle position, velocity, and acceleration from GNSS (GPS) and IMU data. It is a standalone Python script and does not require ROS.

## How It Works
- Reads GNSS and IMU data from CSV files.
- Synchronizes the two streams by timestamp.
- Runs a Kalman filter to estimate:
  - Latitude, longitude, altitude
  - Velocity (in latitude and longitude directions)
  - Acceleration (in latitude and longitude directions)
- Writes the estimated states to a new CSV file.

## Input Data

### 1. GNSS Data (`new_data/gnss.csv`)
A CSV file with the following columns:

| timestamp | lat      | lon      | alt      |
|-----------|----------|----------|----------|
| float     | float    | float    | float    |

- `timestamp`: Time in seconds (float)
- `lat`: Latitude (float)
- `lon`: Longitude (float)
- `alt`: Altitude (float)

### 2. IMU Data (`new_data/imu.csv`)
A CSV file with the following columns:

| timestamp | accel_x  | accel_y  | accel_z  | gyro_x  | gyro_y  | gyro_z  |
|-----------|----------|----------|----------|---------|---------|---------|
| float     | float    | float    | float    | float   | float   | float   |

- `timestamp`: Time in seconds (float)
- `accel_x`, `accel_y`, `accel_z`: Linear acceleration (float)
- `gyro_x`, `gyro_y`, `gyro_z`: Angular velocity (float)

## Output Data

### Processed Sensor Data (`new_data/processed_sensor_data.csv`)
A CSV file with the following columns:

| timestamp | lat      | lon      | altitude | velocity_lat | velocity_lon | accel_lat | accel_lon |
|-----------|----------|----------|----------|--------------|--------------|-----------|-----------|
| float     | float    | float    | float    | float        | float        | float     | float     |

- `timestamp`: Time in seconds (float)
- `lat`, `lon`, `altitude`: Estimated position
- `velocity_lat`, `velocity_lon`: Estimated velocity
- `accel_lat`, `accel_lon`: Estimated acceleration

## Usage
1. Place your input files in the `new_data/` directory.
2. Run the script:
   ```
   python3 state_estimation.py
   ```
3. The output will be saved as `new_data/processed_sensor_data.csv`. 
