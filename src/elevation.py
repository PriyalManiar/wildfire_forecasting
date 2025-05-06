import numpy as np
import requests
import time

# Define bounding box (example)
min_lat, max_lat = 32.5, 33.5
min_lon, max_lon = -117.6, -116.1
GRID_SIZE = 25

# Generate the lat/lon grid
lat_grid = np.linspace(min_lat, max_lat, GRID_SIZE)
lon_grid = np.linspace(min_lon, max_lon, GRID_SIZE)

# Create location strings for API
locations = []
for lat in lat_grid:
    for lon in lon_grid:
        locations.append(f"{lat:.5f},{lon:.5f}")

# Query in batches of 100 (API limit)
elevation_grid = []
batch_size = 100
dataset = "etopo1"  # Recommended real dataset

for i in range(0, len(locations), batch_size):
    batch = "|".join(locations[i:i + batch_size])
    url = f"https://api.opentopodata.org/v1/{dataset}?locations={batch}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for result in data["results"]:
            elevation_grid.append(result["elevation"])
    else:
        print("API error:", response.status_code)
    time.sleep(1)  # Be polite and avoid rate limits

# Reshape into 2D grid
elevation_array = np.array(elevation_grid).reshape(len(lat_grid), len(lon_grid))

# Optional: Display elevation map
import matplotlib.pyplot as plt
plt.imshow(elevation_array, cmap='terrain', origin='lower')
plt.colorbar(label='Elevation (m)')
plt.title("Elevation Grid")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.show()


print("\nCross-validating sample grid points with OpenTopoData...\n")

# Choose 5 key points (row, col)
validation_points = {
    "bottom_left": (0, 0),
    "top_right": (GRID_SIZE - 1, GRID_SIZE - 1),
    "center": (GRID_SIZE // 2, GRID_SIZE // 2),
    "mid_left": (GRID_SIZE // 2, 0),
    "mid_bottom": (0, GRID_SIZE // 2)
}

for label, (i, j) in validation_points.items():
    lat = lat_grid[i]
    lon = lon_grid[j]
    grid_value = elevation_array[i, j]

    # Query OpenTopoData again
    check_url = f"https://api.opentopodata.org/v1/{dataset}?locations={lat:.5f},{lon:.5f}"
    check_resp = requests.get(check_url)
    if check_resp.status_code == 200:
        actual = check_resp.json()['results'][0]['elevation']
        diff = abs(actual - grid_value)
        print(f"{label.upper()} ({lat:.5f}, {lon:.5f}):")
        print(f"  Grid Elevation:    {grid_value:.2f} m")
        print(f"  API Elevation:     {actual:.2f} m")
        print(f"  Difference:        {diff:.2f} m\n")
    else:
        print(f"{label.upper()}: API error: {check_resp.status_code}")