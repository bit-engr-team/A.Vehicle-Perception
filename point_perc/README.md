# LiDAR Moving Object Detection and Collision Prediction

This project provides a system that detects moving objects using LiDAR data and predicts potential collisions between the robot and these objects. The system clusters objects, tracks their movement, and simulates potential collision scenarios with the robot.

## Features

- **LiDAR Data Cleaning**: Removes noise and outliers from the LiDAR data.
- **Object Clustering**: Clusters moving objects using the DBSCAN algorithm.
- **Object Tracking**: Tracks objects by comparing their previous and current positions.
- **Collision Prediction**: Predicts potential collisions by analyzing the movement of both the robot and objects.
- **Data Visualization**: Visualizes the objects, robot's movement, and potential collisions in 3D.

## Algorithms and Models Used

- **Statistical Outlier Removal (SOR)**: Cleans noise and outliers from LiDAR data.
- **DBSCAN**: A density-based clustering algorithm that groups objects based on their proximity.
- **Nearest Neighbors**: Used for matching previous and current object centroids by calculating distances.
- **Unscented Kalman Filter (UKF)**: Used to model and predict the motion of objects.
- **Collision Prediction**: Predicts potential collisions by calculating relative velocities and distances between the robot and objects.

## Requirements

- `open3d`: For LiDAR data processing and 3D visualization.
- `numpy`: For mathematical calculations.
- `matplotlib`: For color visualizations.
- `sklearn`: For DBSCAN clustering and nearest neighbor calculations.

## Inspiration

This project is inspired by the [Point Clouds 3D Perception with Open3D](https://github.com/yudhisteer/Point-Clouds-3D-Perception-with-Open3D?tab=readme-ov-file#3bb). You can check out the project for more details.


You can install the required packages with the following command:

```bash
pip install open3d numpy matplotlib scikit-learn
