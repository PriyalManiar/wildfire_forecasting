import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Grid boundaries and resolution (match fire_grid_analysis)
min_lat, max_lat = 32.5, 33.5
min_lon, max_lon = -117.6, -116.1
GRID_SIZE = 25

# Read humidity data
df = pd.read_csv('monthly_averages.csv')
humidity_values = df['MonthlyAverageRelativeHumidity'].dropna().values

# Create humidity grid by random sampling
humidity_grid = np.random.choice(humidity_values, size=(GRID_SIZE, GRID_SIZE))

# Save grid
np.save('humidity_grid_25x25.npy', humidity_grid)

# Plot
plt.figure(figsize=(7, 6))
plt.imshow(humidity_grid, cmap='Blues', origin='lower', extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
plt.colorbar(label='Relative Humidity (%)')
plt.title('Randomly Sampled Humidity Grid (25x25)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('humidity_grid_25x25.png')
plt.close()
print('Saved: humidity_grid_25x25.png')