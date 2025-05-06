import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Grid boundaries and resolution (from fire_grid_analysis)
min_lat, max_lat = 32.5, 33.5
min_lon, max_lon = -117.6, -116.1
GRID_SIZE = 25
lat_grid = np.linspace(min_lat, max_lat, GRID_SIZE + 1)
lon_grid = np.linspace(min_lon, max_lon, GRID_SIZE + 1)

# Ignition probabilities for each vegetation type
ignition_probs = {
    "Openshrub": 0.4,
    "C3_grass": 0.6,
    "C3past": 0.3,
    "DenseShrub": 0.5,
    "SecTmpENF": 0.3,
    "tmpENF": 0.6,
    "urban": 0.2,
    "water": 0.0
}

veg_folder = 'vegetation'
veg_files = [
    'Openshrub.csv',
    'C3_grass.csv',
    'C3past.csv',
    'DenseShrub.csv',
    'SecTmpENF.csv',
    'tmpENF.csv',
    'urban.csv',
    'water.csv',
]
veg_keys = [
    'Openshrub',
    'C3_grass',
    'C3past',
    'DenseShrub',
    'SecTmpENF',
    'tmpENF',
    'urban',
    'water',
]

# Initialize the ignition probability grid
ignition_grid = np.zeros((GRID_SIZE, GRID_SIZE))

for veg_file, veg_key in zip(veg_files, veg_keys):
    veg_path = os.path.join(veg_folder, veg_file)
    df = pd.read_csv(veg_path)
    veg_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for _, row in df.iterrows():
        try:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            perc = float(row['percentage'])
        except (ValueError, TypeError):
            continue
        if lon > 180:
            lon = lon - 360
        if pd.isna(perc):
            continue
        i = np.digitize(lat, lat_grid) - 1
        j = np.digitize(lon, lon_grid) - 1
        if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
            veg_grid[i, j] = perc
    ignition_grid += (veg_grid / 100.0) * ignition_probs[veg_key]

# Plot the final ignition probability grid
plt.figure(figsize=(7, 6))
plt.imshow(ignition_grid, cmap='OrRd', origin='lower', extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Aggregated Ignition Probability')
plt.title('Aggregated Ignition Probability (25x25 Grid)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('ignition_probability_25x25.png')
plt.close()
print('Saved: ignition_probability_25x25.png') 