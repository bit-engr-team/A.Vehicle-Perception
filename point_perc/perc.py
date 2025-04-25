import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Clean the point cloud by removing outliers
def clean_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

# Assign a unique color based on the cluster index
def assign_unique_color(index):
    cmap = plt.get_cmap("tab20")
    color = cmap(index % cmap.N)[:3]
    return list(color)

# Cluster the point cloud data using DBSCAN
def cluster_objects(pcd, eps=0.5, min_points=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = {}
    for idx in np.unique(labels):
        if idx == -1:  # Skip noise points
            continue
        indices = np.where(labels == idx)[0]
        cluster_points = np.asarray(pcd.points)[indices]
        clusters[idx] = cluster_points
    return clusters

# Compute the centroids of each cluster
def compute_centroids(clusters):
    centroids = []
    for pts in clusters.values():
        centroids.append(np.mean(pts, axis=0))
    return centroids

# Match objects from previous and current frames based on centroids
def match_objects(previous_objects, current_centroids):
    object_assignments = {}
    if not previous_objects:
        for i, c in enumerate(current_centroids):
            object_assignments[i] = i
        return object_assignments

    previous_centroids = np.array([v[-1] for v in previous_objects.values()])
    if len(previous_centroids) == 0 or len(current_centroids) == 0:
        return object_assignments

    current_centroids = np.array(current_centroids)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(previous_centroids)
    distances, indices = nbrs.kneighbors(current_centroids)

    used = set()
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx[0] not in used:
            object_assignments[i] = idx[0]
            used.add(idx[0])
    return object_assignments

# UKF prediction step to predict object state and covariance
def ukf_predict(state, covariance, process_noise, dt=1):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    state = np.dot(F, state)
    covariance = np.dot(np.dot(F, covariance), F.T) + process_noise

    return state, covariance

# Compute time to collision (TTC) between robot and object
def compute_time_to_collision(robot_position, robot_velocity, object_position, object_velocity):
    relative_position = np.array(object_position) - np.array(robot_position)
    relative_velocity = np.array(object_velocity) - np.array(robot_velocity)

    relative_velocity_magnitude = np.linalg.norm(relative_velocity)
    if relative_velocity_magnitude == 0:
        return np.inf

    ttc = -np.dot(relative_position, relative_velocity) / (relative_velocity_magnitude ** 2)
    return ttc if ttc >= 0 else np.inf

# Predict potential collisions between robot and objects
def predict_collisions(object_paths, robot_path, dt=1.0, collision_distance=1.0):
    if len(robot_path) < 2:
        return [], []

    robot_velocity = np.array(robot_path[-1]) - np.array(robot_path[-2])
    robot_position = np.array(robot_path[-1])

    collisions = []

    for obj_id, path in object_paths.items():
        if len(path) < 2:
            continue

        object_velocity = np.array(path[-1]) - np.array(path[-2])
        object_position = np.array(path[-1])

        ttc = compute_time_to_collision(robot_position, robot_velocity, object_position, object_velocity)

        if ttc <= dt and np.linalg.norm(object_position - robot_position) <= collision_distance:
            collisions.append(obj_id)
            print(f"Çarptı! Nesne {obj_id} ile çarpışma gerçekleşti!")  # Collision detected message

    return collisions, []

# Determine the relative direction of an object with respect to the robot
def get_direction(object_pos, robot_pos):
    diff = np.array(object_pos) - np.array(robot_pos)
    angle = np.arctan2(diff[1], diff[0])

    if -np.pi/4 <= angle < np.pi/4:
        return "sağda"
    elif np.pi/4 <= angle < 3*np.pi/4:
        return "önde"
    elif angle >= 3*np.pi/4 or angle < -3*np.pi/4:
        return "solda"
    else:
        return "arkada"

# Visualize the LiDAR data, robot, and objects
def visualize(vis, pcd_static, pcd_dynamic, object_paths, robot_path, collision_ids, near_objects):
    vis.clear_geometries()

    # Clean and visualize the static point cloud
    pcd_static = clean_point_cloud(pcd_static)
    pcd_static.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pcd_static)

    # Visualize the dynamic point cloud
    vis.add_geometry(pcd_dynamic)

    # Visualize object paths
    for idx, (obj_id, path) in enumerate(object_paths.items()):
        color = [1.0, 0, 0] if obj_id in collision_ids else assign_unique_color(idx)
        obj_path = o3d.geometry.PointCloud()
        obj_path.points = o3d.utility.Vector3dVector(np.array(path))
        obj_path.paint_uniform_color(color)
        vis.add_geometry(obj_path)

    # Visualize the robot path
    if len(robot_path) >= 2:
        lines = [[i, i + 1] for i in range(len(robot_path) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(robot_path))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in lines])
        vis.add_geometry(line_set)

    # Print nearby object messages
    for obj_id, msg in near_objects:
        print(f"Nesne {obj_id}: {msg}")

# Load all point clouds from a folder
def load_point_clouds_from_folder(folder_path):
    pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])
    return [o3d.io.read_point_cloud(os.path.join(folder_path, f)) for f in pcd_files]

# Main program for LiDAR simulation
if __name__ == "__main__":
    folder_path = r"D:\bit\exp22-kazali\exp22\lidar"  # Path to your data folder
    pcd_sequence = load_point_clouds_from_folder(folder_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Simülasyonu", width=1280, height=720)
    vis.get_render_option().background_color = np.array([0, 0, 0])

    object_paths = {}
    robot_path = []
    frames = []

    previous_objects = {}

    # Process each LiDAR frame in sequence
    for i in range(1, len(pcd_sequence)):
        pcd_prev = pcd_sequence[i - 1]
        pcd_curr = pcd_sequence[i]

        # Cluster current frame and compute centroids
        clusters = cluster_objects(pcd_curr)
        centroids = compute_centroids(clusters)

        # Match objects between previous and current frames
        assignments = match_objects(previous_objects, centroids)

        updated_objects = {}
        for curr_idx, assigned_id in assignments.items():
            pos = centroids[curr_idx]
            if assigned_id in object_paths:
                object_paths[assigned_id].append(pos)
            else:
                object_paths[assigned_id] = [pos]
            updated_objects[assigned_id] = object_paths[assigned_id]

        previous_objects = updated_objects

        # Update robot position and path
        robot_pos = np.mean(np.asarray(pcd_curr.points), axis=0)
        robot_path.append(robot_pos)

        # Check for potential collisions
        collision_ids, near_objects = predict_collisions(object_paths, robot_path)

        # Visualize the data
        visualize(vis, pcd_prev, pcd_curr, object_paths, robot_path, collision_ids, near_objects)
        vis.poll_events()
        vis.update_renderer()

    # Run the visualizer window
    vis.run()
    vis.destroy_window()
