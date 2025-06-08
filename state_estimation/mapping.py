#!/usr/bin/env python3
"""
Map trajectory plotter: Reads latitude and longitude from processed_sensor_data.csv and saves a map image.
"""
import csv
import matplotlib.pyplot as plt

INPUT_CSV = 'new_data/processed_sensor_data.csv'
OUTPUT_IMG = 'new_data/psd.png'

lats = []
lons = []

with open(INPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lats.append(float(row['lat']))
        lons.append(float(row['lon']))

plt.figure(figsize=(10, 8))
plt.plot(lons, lats, marker='o', markersize=1, linewidth=1, label='Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Estimated Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig(OUTPUT_IMG)
print(f"Trajectory map saved to {OUTPUT_IMG}")

# Note: Install matplotlib if needed: pip install matplotlib
