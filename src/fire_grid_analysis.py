import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('NASA_FIRMS.txt')

# Step 1: Confidence Filtering (only 'nominal' or 'high')
df = df[df['confidence'].isin(['nominal', 'high'])]

# Focus on Southeast US bounding box
region_mask = (
    (df['latitude'] >= 33) & (df['latitude'] <= 36) &
    (df['longitude'] >= -81) & (df['longitude'] <= -78)
)
region_data = df[region_mask].copy()

# Automatically find the tightest bounding box around detected fires in this region
min_lat = region_data['latitude'].min()
max_lat = region_data['latitude'].max()
min_lon = region_data['longitude'].min()
max_lon = region_data['longitude'].max()

GRID_SIZE = 25
lat_grid = np.linspace(min_lat, max_lat, GRID_SIZE + 1)
lon_grid = np.linspace(min_lon, max_lon, GRID_SIZE + 1)

region_data['lat_bin'] = np.digitize(region_data['latitude'], lat_grid) - 1
region_data['lon_bin'] = np.digitize(region_data['longitude'], lon_grid) - 1
region_data['cell_id'] = region_data['lat_bin'] * GRID_SIZE + region_data['lon_bin']

# Step 3: Deduplicate by cell and date (unique fire events)
region_data['acq_date'] = pd.to_datetime(region_data['acq_date'])
unique_events = region_data.drop_duplicates(subset=['cell_id', 'acq_date'])

# Step 4: Count-based probability only
fire_counts = np.zeros((GRID_SIZE, GRID_SIZE))
for _, row in region_data.iterrows():
    i, j = int(row['lat_bin']), int(row['lon_bin'])
    if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
        fire_counts[i, j] += 1

total_fires = fire_counts.sum()
prob_counts = fire_counts / total_fires if total_fires > 0 else fire_counts

# Save output
np.save('fire_probabilities_counts.npy', prob_counts)

# Visualization: Only count-based probability and scatter plot
plt.figure(figsize=(14, 6))

# Heatmap
plt.subplot(1, 2, 1)
plt.imshow(prob_counts, cmap='hot', origin='lower', extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
plt.colorbar(label='Probability (Count-based)')
plt.title('Fire Probability (Count-based, 25x25 Grid, Southeast US)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(region_data['longitude'], region_data['latitude'], c='red', s=20, alpha=0.7, label='Fire Detections')
plt.title('Fire Detections (Scatter)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(min_lon, max_lon)
plt.ylim(min_lat, max_lat)
plt.legend()

plt.tight_layout()
plt.savefig('fire_probability_and_scatter_25x25_southeast_us.png')
plt.close()

# Calculate cell sizes in kilometers
lat_cell_size_km = (max_lat - min_lat) * 111 / GRID_SIZE  # 1 degree ≈ 111 km
lon_cell_size_km = (max_lon - min_lon) * 111 * np.cos(np.radians((min_lat+max_lat)/2)) / GRID_SIZE

# Print summary statistics
print(f"Southeast US bounding box:")
print(f"Latitude: {min_lat:.4f} to {max_lat:.4f}")
print(f"Longitude: {min_lon:.4f} to {max_lon:.4f}")
print(f"Cell size in latitude: {lat_cell_size_km:.2f} km")
print(f"Cell size in longitude: {lon_cell_size_km:.2f} km")
print(f"Total fire detections: {total_fires}")
print(f"Grid cells with fires: {np.count_nonzero(fire_counts)}")
print(f"Average fires per cell (non-zero): {fire_counts[fire_counts > 0].mean():.2f}")
print(f"Fire density: {total_fires/(GRID_SIZE*GRID_SIZE*lat_cell_size_km*lon_cell_size_km):.4f} fires/km²")

# Calculate binary fire occurrence (presence/absence)
fire_presence = (fire_counts > 0).astype(int)
np.save('fire_presence_25x25_southeast_us.npy', fire_presence)

# Visualize the presence/absence map
plt.figure(figsize=(7, 6))
plt.imshow(fire_presence, cmap='Greens', origin='lower', extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
plt.colorbar(label='Fire Occurrence (1=Occurred, 0=Never)')
plt.title('Fire Occurrence (Presence/Absence, 25x25 Grid, Southeast US)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('fire_presence_25x25_southeast_us.png')
plt.close()

print(f"Number of cells where fire ever occurred: {fire_presence.sum()} out of {GRID_SIZE*GRID_SIZE}") 